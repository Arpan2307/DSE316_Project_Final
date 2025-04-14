
import torch.nn as nn

class FeatureCalibrationNet(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(256)  # Reduce the size of input to 256
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        # Apply the pooling layer before passing to Linear layers
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return self.net(x)

