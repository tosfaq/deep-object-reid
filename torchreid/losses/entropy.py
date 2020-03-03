import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0.0, p * p.log(), torch.zeros_like(p)).sum(dim=dim, keepdim=keepdim)


class EntropyLoss(nn.Module):
    def __init__(self, scale=1.0):
        super(EntropyLoss, self).__init__()

        self.scale = scale
        assert self.scale > 0.0

    def forward(self, cos_theta):
        probs = F.softmax(self.scale * cos_theta, dim=-1)
        entropy_values = entropy(probs, dim=-1)

        losses = self._calc_losses(cos_theta, entropy_values)

        return losses.mean() if losses.numel() > 0 else losses.sum()

    def _calc_losses(self, cos_theta, entropy_values):
        raise NotImplementedError


class MinEntropyLoss(EntropyLoss):
    def __init__(self, scale=1.0):
        super(MinEntropyLoss, self).__init__(scale)

    def _calc_losses(self, cos_theta, entropy_values):
        return entropy_values


class MaxEntropyLoss(EntropyLoss):
    def __init__(self, scale=1.0):
        super(MaxEntropyLoss, self).__init__(scale)

    def _calc_losses(self, cos_theta, entropy_values):
        return np.log(cos_theta.size(-1)) - entropy_values
