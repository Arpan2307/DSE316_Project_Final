import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchaudio.datasets import LIBRISPEECH
from torchaudio.transforms import Resample
from model.feature_extractor import AudioFeatureExtractor
from model.calibration_net import FeatureCalibrationNet
from utils.losses import contrastive_loss, knowledge_distillation_loss
from utils.prototype_memory import PrototypeMemory
import torchaudio
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset (single stage example)
os.makedirs(LIBRISPEECH_PATH, exist_ok=True)

train_set = LIBRISPEECH(LIBRISPEECH_PATH, url="train-clean-100", download=True)
resampler = Resample(orig_freq=16000, new_freq=16000)

# Collate function that handles batching and padding
def collate_fn(batch):
    waveforms, labels = [], []

    # Find max length of waveform in the batch
    max_length = max([waveform.shape[1] for waveform, *_ in batch])

    for waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id in batch:
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad the waveform to max_length
        padding = max_length - waveform.shape[1]
        padded_waveform = torch.nn.functional.pad(waveform, (0, padding))

        waveforms.append(padded_waveform)
        labels.append(hash(speaker_id) % (10 ** 8))  # Use a hash of the speaker_id for unique numeric labels

    return torch.stack(waveforms), torch.tensor(labels)

# Initialize data loader
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Map labels (speaker_ids) to class indices
def map_labels_to_classes(labels):
    unique_labels = torch.unique(labels)
    label_to_class = {label.item(): idx for idx, label in enumerate(unique_labels)}
    mapped_labels = torch.tensor([label_to_class[label.item()] for label in labels], device=device)
    return mapped_labels, len(unique_labels)

# Model setup
model = FeatureCalibrationNet().to(device)  # Replace with your actual model

# Loss functions
ce_loss = nn.CrossEntropyLoss()

# Optimizer setup
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()  # Set model to training mode
    total_loss = 0

    for batch_idx, (waveforms, labels) in enumerate(train_loader):
        waveforms, labels = waveforms.to(device), labels.to(device)

        # Map the labels to class indices
        mapped_labels, n_classes = map_labels_to_classes(labels)

        # Forward pass
        logits = model(waveforms)  # Replace with actual model's forward pass
        
        # Compute cross-entropy loss
        loss_ce = ce_loss(logits, mapped_labels)
        
        # Optionally add other losses (e.g., contrastive_loss, knowledge_distillation_loss)
        loss = loss_ce  # Combine losses if necessary

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print epoch statistics
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(train_loader)}")

# Save model checkpoint if necessary
torch.save(model.state_dict(), "model.pth")

