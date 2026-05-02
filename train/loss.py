import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # This should be a tensor (or None)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
        else:
            alpha = None

        ce_loss = F.cross_entropy(inputs, targets, weight=alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def get_loss(counts = [3616, 6012, 10192, 1345]):
    class_counts = torch.tensor(counts, dtype=torch.float)
    alpha = 1.0 / class_counts
    alpha = alpha / alpha.sum()  # Normalize
    return FocalLoss(alpha=alpha, gamma=2.0)

