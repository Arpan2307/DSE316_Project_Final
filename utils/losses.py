
import torch
import torch.nn.functional as F

def contrastive_loss(anchor, positive, negative, margin=1.0):
    d_pos = F.pairwise_distance(anchor, positive)
    d_neg = F.pairwise_distance(anchor, negative)
    return torch.mean(F.relu(d_pos - d_neg + margin))

def knowledge_distillation_loss(old_feats, new_feats):
    return F.mse_loss(new_feats, old_feats)
