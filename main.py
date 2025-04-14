
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

def collate_fn(batch):
    waveforms, labels = [], []
    for waveform, _, speaker_id, *_ in batch:
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveforms.append(resampler(waveform))
        labels.append(speaker_id)
    return torch.cat(waveforms), torch.tensor(labels)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)

# Initialize models
feature_extractor = AudioFeatureExtractor(FEATURE_DIM).to(device)
classifier = nn.Linear(FEATURE_DIM, NUM_CLASSES).to(device)
calibration_net = FeatureCalibrationNet(FEATURE_DIM).to(device)

optimizer = optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()) + list(calibration_net.parameters()), lr=LEARNING_RATE)
ce_loss = nn.CrossEntropyLoss()

# Prototype Memory
proto_memory = PrototypeMemory(FEATURE_DIM, device)

# Placeholder for old model features
old_feature_extractor = None

for epoch in range(EPOCHS):
    feature_extractor.train()
    classifier.train()
    for waveforms, labels in train_loader:
        waveforms, labels = waveforms.to(device), labels.to(device)

        features = feature_extractor(waveforms)
        logits = classifier(features)
        loss_ce = ce_loss(logits, labels)

        loss_kd = torch.tensor(0.0).to(device)
        loss_co = torch.tensor(0.0).to(device)

        if old_feature_extractor:
            with torch.no_grad():
                old_feats = old_feature_extractor(waveforms)
            new_feats_calibrated = calibration_net(features)
            loss_kd = knowledge_distillation_loss(old_feats, new_feats_calibrated)

            proto_feats, proto_labels = proto_memory.get_all()
            pos_mask = labels.unsqueeze(1) == proto_labels.unsqueeze(0)
            for i in range(features.size(0)):
                positives = proto_feats[pos_mask[i]]
                negatives = proto_feats[~pos_mask[i]]
                if positives.size(0) > 0 and negatives.size(0) > 0:
                    pos_feat = positives[0].unsqueeze(0)
                    neg_feat = negatives[0].unsqueeze(0)
                    loss_co += contrastive_loss(features[i].unsqueeze(0), pos_feat, neg_feat)
            loss_co /= features.size(0)

        total_loss = loss_ce + 0.5 * loss_kd + 0.1 * loss_co
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: CE={loss_ce.item():.4f}, KD={loss_kd.item():.4f}, CO={loss_co.item():.4f}")

# Save current model and prototypes
proto_memory.update(features, labels)
torch.save(feature_extractor.state_dict(), "new_model.pth")
torch.save(classifier.state_dict(), "classifier.pth")
