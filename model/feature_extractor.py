import torch
import torch.nn as nn
import torchaudio

class AudioFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec = bundle.get_model()

        # To determine the output dimension, we will pass a dummy tensor through the model
        # Create a dummy input (batch size 1, 16000 samples) to check the output shape
        dummy_input = torch.zeros(1, 16000)
        with torch.no_grad():
            output, _ = self.wav2vec.extract_features(dummy_input)

        # The output of `extract_features` is a list of feature tensors; we take the last one
        output_dim = output[-1].size(-1)  # Last dimension of the last feature tensor
        self.project = nn.Linear(output_dim, feature_dim)

    def forward(self, x):
        with torch.no_grad():
            feats, _ = self.wav2vec.extract_features(x)
        x = self.project(feats[-1].mean(dim=1))  # Average pooling
        return x

