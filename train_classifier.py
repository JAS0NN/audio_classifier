# train_classifier.py

import os
import torch
import pandas as pd
from datasets import load_dataset, Audio, Dataset
from transformers import (Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification,
                          TrainingArguments, Trainer, EvalPrediction)
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# =======================
# Parameters
# =======================
data_csv = "data.csv"  # should be generated from your dataset folder
model_checkpoint = "facebook/wav2vec2-large-xlsr-53"
output_dir = "./language_classifier"
label2id = {"zh": 0, "taigi": 1, "cs": 2}
id2label = {v: k for k, v in label2id.items()}

# =======================
# Load and preprocess dataset
# =======================
df = pd.read_csv(data_csv)
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

processor = Wav2Vec2FeatureExtractor.from_pretrained(model_checkpoint)

@dataclass
class DataCollatorForClassification:
    """
    Simple data collator for fixed-length audio classification.
    """
    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, torch.Tensor]:
        # 所有音檔都是固定長度，直接堆疊
        input_values = [feature["input_values"] for feature in features]
        labels = [feature["label"] for feature in features]
        
        batch = {
            "input_values": torch.tensor(np.array(input_values), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
        
        return batch

data_collator = DataCollatorForClassification()

def preprocess(example):
    audio = example["path"]
    audio_array = audio["array"]
    
    # 固定所有音檔長度為10秒
    target_length = 160000  # 10秒 * 16000 = 160000 samples
    
    if len(audio_array) > target_length:
        # 隨機裁切
        start_idx = np.random.randint(0, len(audio_array) - target_length + 1)
        audio_array = audio_array[start_idx:start_idx + target_length]
    elif len(audio_array) < target_length:
        # 補零padding
        pad_length = target_length - len(audio_array)
        audio_array = np.pad(audio_array, (0, pad_length), mode='constant', constant_values=0)
    
    # 處理音訊，返回固定長度
    inputs = processor(
        audio_array, 
        sampling_rate=16000, 
        return_tensors="np"
    )
    inputs["label"] = label2id[example["label"]]
    
    # 確保input_values是正確的形狀
    inputs["input_values"] = inputs["input_values"].flatten()
    
    return inputs

# =======================
# Load model
# =======================
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)

# =======================
# Train/test split
# =======================
dataset = dataset.train_test_split(test_size=0.1, seed=42)
dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

# =======================
# Metrics
# =======================
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# =======================
# TrainingArguments & Trainer
# =======================
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",  # 修復 evaluation_strategy 已被棄用的警告
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    logging_dir=f"{output_dir}/logs",
    logging_steps=10,
    dataloader_num_workers=0,  # 避免多進程問題
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,  # 使用自定義的data collator
    compute_metrics=compute_metrics,
)

# =======================
# Train and Evaluate
# =======================
trainer.train()
trainer.save_model(output_dir)

# =======================
# Evaluation
# =======================
preds_output = trainer.predict(dataset["test"])
y_pred = np.argmax(preds_output.predictions, axis=1)
y_true = preds_output.label_ids
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=list(label2id.keys())))
