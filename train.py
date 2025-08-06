
import torch
from torch.utils.data import random_split
from transformers import WhisperFeatureExtractor, TrainingArguments, Trainer
from dataset import AudioDataset
from model import AudioClassifier
import numpy as np
from sklearn.metrics import accuracy_score

# Configuration
DATA_DIR = "dataset"
MODEL_NAME = "openai/whisper-base"

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

def main():
    # Initialize feature extractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)

    # Create dataset
    dataset = AudioDataset(data_dir=DATA_DIR, feature_extractor=feature_extractor)

    # Optional: use only a subset of the data (e.g., 10%) for faster experiments
    subset_ratio = 0.1
    subset_size = max(1, int(len(dataset) * subset_ratio))

    # Ensure at least 1 sample in each split when subset is small
    train_size = max(1, int(subset_size * 0.8))
    val_size = max(1, subset_size - train_size)

    # Get a deterministic subset using a fixed generator seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    full_indices = torch.randperm(len(dataset), generator=generator)[:subset_size]
    train_indices = full_indices[:train_size]
    val_indices = full_indices[train_size:train_size + val_size]

    # Create subset datasets
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices.tolist())
    val_dataset = Subset(dataset, val_indices.tolist())

    # Initialize model
    model = AudioClassifier(model_name=MODEL_NAME)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,  # Reduced for a quicker test run
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",   # transformers>=4.55 uses eval_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model("./best_model_trainer")

if __name__ == "__main__":
    main()

