
import torch
import torch.nn as nn
from transformers import WhisperModel

class AudioClassifier(nn.Module):
    def __init__(self, model_name="openai/whisper-base", num_labels=3):
        super(AudioClassifier, self).__init__()
        self.whisper = WhisperModel.from_pretrained(model_name)
        self.whisper.encoder.requires_grad_(False)  # Freeze encoder
        self.classifier = nn.Sequential(
            nn.Linear(self.whisper.config.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_features, labels=None):
        # Whisper encoder expects input_features of shape (batch, feature_len, feature_dim) or similar;
        # dataset provides input_features as a tensor; ensure batch dimension exists.
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)

        outputs = self.whisper.encoder(input_features)
        hidden_states = outputs.last_hidden_state  # (batch, seq, hidden)

        # Mean pooling over time dimension
        pooled_output = torch.mean(hidden_states, dim=1)  # (batch, hidden)

        logits = self.classifier(pooled_output)  # (batch, num_labels)

        if labels is not None:
            # labels expected shape: (batch,)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


