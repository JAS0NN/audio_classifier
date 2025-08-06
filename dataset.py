
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperFeatureExtractor
import os
import librosa

class AudioDataset(Dataset):
    def __init__(self, data_dir, feature_extractor, max_length=30):
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        self.max_length_seconds = max_length
        self.file_paths = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        class_map = {"taigi": 0, "zh": 1, "cs": 2}
        for class_name, label in class_map.items():
            class_dir = os.path.join(self.data_dir, class_name)
            for file_name in os.listdir(class_dir):
                self.file_paths.append(os.path.join(class_dir, file_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load and resample audio
        waveform, sample_rate = librosa.load(file_path, sr=16000)

        # Truncate or pad audio
        target_length = self.max_length_seconds * sample_rate
        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        else:
            waveform = librosa.util.pad_center(waveform, size=target_length)


        # Extract features
        features = self.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
        input_features = features.input_features.squeeze(0)

        return {"input_features": input_features, "labels": torch.tensor(label)}


