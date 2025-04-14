
import torch

class PrototypeMemory:
    def __init__(self, feat_dim, device):
        self.device = device
        self.feat_dim = feat_dim
        self.prototypes = {}

    def update(self, features, labels):
        for feat, label in zip(features, labels):
            label = label.item()
            if label not in self.prototypes:
                self.prototypes[label] = []
            self.prototypes[label].append(feat.detach().cpu())
        for label in self.prototypes:
            self.prototypes[label] = torch.stack(self.prototypes[label]).mean(dim=0)

    def get_all(self):
        all_feats, all_labels = [], []
        for label, feat in self.prototypes.items():
            all_feats.append(feat.to(self.device))
            all_labels.append(torch.tensor(label, device=self.device))
        return torch.stack(all_feats), torch.stack(all_labels)
