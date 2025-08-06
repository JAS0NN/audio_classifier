# train_classifier_whisper.py

import os
import torch
import pandas as pd
from datasets import load_dataset, Audio, Dataset
from transformers import (
    WhisperFeatureExtractor, WhisperModel, TrainingArguments, Trainer, EvalPrediction
)
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# =======================
# 參數設定
# =======================
data_csv = "data.csv"  # 與原流程相同
# 建議使用 openai/whisper-tiny.en（HuggingFace 官方有 preprocessor_config.json）
model_checkpoint = "openai/whisper-tiny.en"
output_dir = "./language_classifier_whisper"
label2id = {"zh": 0, "taigi": 1, "cs": 2}
id2label = {v: k for k, v in label2id.items()}

# =======================
# Whisper 分類模型
# =======================
class WhisperForAudioClassification(torch.nn.Module):
    """
    Whisper 音訊分類模型

    以 Whisper encoder 為 backbone，接上 dropout 與線性分類 head。
    支援 freeze_whisper 參數凍結 encoder 權重。

    forward(input_features, labels=None)：
        - input_features: (batch, n_mels, n_frames)
        - labels: (batch,) or None
        回傳 (loss, logits, hidden_states)
    """
    def __init__(self, model_name, num_labels, freeze_whisper=True, dropout_prob=0.1):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(model_name)
        print(f"[DEBUG] self.whisper type: {type(self.whisper)}")
        self.dropout = torch.nn.Dropout(dropout_prob)
        hidden_size = self.whisper.config.d_model
        self.classifier = torch.nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

        if freeze_whisper:
            for param in self.whisper.parameters():
                param.requires_grad = False

    def forward(self, input_features, labels=None):
        """
        前向傳播

        Args:
            input_features (torch.FloatTensor): (batch, n_mels, n_frames)
            labels (torch.LongTensor, optional): (batch,)

        Returns:
            loss, logits, hidden_states
        """
        # 只用 encoder，不經過 forward 以避免 decoder_input_ids 問題
        encoder_outputs = self.whisper.encoder(input_features)
        hidden_states = encoder_outputs.last_hidden_state  # (batch, n_frames, hidden_size)
        pooled = hidden_states.mean(dim=1)         # (batch, hidden_size)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)

        return (loss, logits, hidden_states)

# =======================
# 資料載入與前處理
# =======================
df = pd.read_csv(data_csv)
# 只使用一小部分資料進行試跑
sample_size = 10  # 每個類別取樣數量
sampled_dfs = []
for label in label2id.keys():
    label_df = df[df['label'] == label]
    if len(label_df) > sample_size:
        sampled_dfs.append(label_df.sample(sample_size, random_state=42))
    else:
        sampled_dfs.append(label_df)
df_small = pd.concat(sampled_dfs)
print(f"原始資料量: {len(df)}，取樣後資料量: {len(df_small)}")
dataset = Dataset.from_pandas(df_small)
# 不使用 dataset.cast_column("path", Audio(sampling_rate=16000))
# 因為它依賴於 torchcodec，而 torchcodec 與 FFmpeg4 有兼容性問題

try:
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_checkpoint)
except OSError as e:
    print(f"WhisperFeatureExtractor 載入失敗，請確認 model_checkpoint='{model_checkpoint}' 是否正確，且 HuggingFace 上該 repo 有 preprocessor_config.json。")
    print("建議使用 openai/whisper-tiny.en、openai/whisper-base、openai/whisper-small 等官方模型。")
    raise e

import librosa

def preprocess(example):
    """
    Whisper 音訊前處理

    - 使用 librosa 載入音訊
    - 將音訊裁切/補零至 30 秒 (480,000 samples)
    - 經 WhisperFeatureExtractor 得 input_features (n_mels, n_frames)
    - 回傳 dict: input_features, label
    """
    # 使用 librosa 替代 torchcodec 載入音訊
    try:
        audio_array, _ = librosa.load(example["path"], sr=16000, mono=True)
    except Exception as e:
        print(f"無法載入音訊檔案 {example['path']}: {e}")
        # 如果載入失敗，創建一個空白音訊
        audio_array = np.zeros(16000 * 5)  # 5 秒靜音
        
    target_length = 16000 * 30  # 30 秒

    if len(audio_array) > target_length:
        # 隨機裁切
        start_idx = np.random.randint(0, len(audio_array) - target_length + 1)
        audio_array = audio_array[start_idx:start_idx + target_length]
    elif len(audio_array) < target_length:
        # 補零 padding
        pad_length = target_length - len(audio_array)
        audio_array = np.pad(audio_array, (0, pad_length), mode='constant', constant_values=0)

    # WhisperFeatureExtractor 輸出 shape: (1, n_mels, n_frames)
    inputs = feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    )
    # squeeze batch 維度並確保是張量
    input_features = inputs["input_features"].squeeze(0)
    # 確保 input_features 是張量
    if not isinstance(input_features, torch.Tensor):
        input_features = torch.tensor(input_features)
    return {
        "input_features": input_features,
        "label": label2id[example["label"]]
    }

# =======================
# DataCollator
# =======================
@dataclass
class DataCollatorForWhisperClassification:
    """
    Whisper 分類用 DataCollator

    直接堆疊 input_features (float32, (n_mels, n_frames)) 及 labels
    """
    def __call__(self, features: List[Dict[str, Union[torch.Tensor, int]]]) -> Dict[str, torch.Tensor]:
        input_features = []
        for f in features:
            # 確保每個 input_feature 是張量
            feat = f["input_features"]
            if not isinstance(feat, torch.Tensor):
                feat = torch.tensor(feat)
            input_features.append(feat)
        
        labels = [f["label"] for f in features]
        
        # 確保所有張量都是相同的數據類型
        input_tensor = torch.stack([feat.float() for feat in input_features])
        label_tensor = torch.tensor(labels, dtype=torch.long)
        
        batch = {
            "input_features": input_tensor,
            "labels": label_tensor
        }
        return batch

data_collator = DataCollatorForWhisperClassification()

# =======================
# 資料集切分與前處理
# =======================
dataset = dataset.train_test_split(test_size=0.1, seed=42)
dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

# =======================
# 模型初始化
# =======================
model = WhisperForAudioClassification(
    model_name=model_checkpoint,
    num_labels=len(label2id),
    freeze_whisper=True
)

# =======================
# 評估指標
# =======================
def compute_metrics(p: EvalPrediction):
    """
    計算分類準確率

    Args:
        p: EvalPrediction

    Returns:
        dict: {"accuracy": ...}
    """
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# =======================
# 自定義訓練循環
# =======================
# 檢查 transformers 版本
import transformers
print(f"Transformers 版本: {transformers.__version__}")

print("設定自定義訓練參數...")
# 訓練參數
batch_size = 4
learning_rate = 3e-5
weight_decay = 0.01
num_epochs = 2
device = torch.device("cpu")  # 強制使用 CPU

# 創建資料載入器
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    dataset["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=0
)

eval_dataloader = DataLoader(
    dataset["test"],
    batch_size=batch_size,
    collate_fn=data_collator,
    num_workers=0
)

# 優化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 確保模型在正確的設備上
model.to(device)

def evaluate(model, dataloader, device):
    """評估函數"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            
            _, logits, _ = model(input_features=input_features)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 計算準確率
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"  評估準確率: {accuracy:.4f}")
    
    # 分類報告
    print("\n  分類報告:\n")
    print(classification_report(all_labels, all_preds, target_names=list(label2id.keys())))
    
    return accuracy

# =======================
# 訓練與評估
# =======================
if __name__ == "__main__":
    print("開始訓練...")
    model.train()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # 將資料移到設備上
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            
            # 前向傳播
            optimizer.zero_grad()
            loss, logits, _ = model(input_features=input_features, labels=labels)
            
            # 反向傳播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"  批次 {batch_idx}/{len(train_dataloader)}, 損失: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"  Epoch {epoch+1} 平均損失: {avg_loss:.4f}")
        
        # 每個 epoch 結束後評估
        print("  進行評估...")
        evaluate(model, eval_dataloader, device)
    
    # 保存模型
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{output_dir}/model.pt")
    print(f"模型已保存到 {output_dir}/model.pt")

    # =======================
    # 測試集分類報告
    # =======================
    print("\n最終測試集評估:")
    evaluate(model, eval_dataloader, device)