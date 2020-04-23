import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TotalVarianceLoss(nn.Module):
    def __init__(self, kernel_size, num_channels, hard_values=True, limits=(0.0, 1.0), threshold=0.5, fraction=0.1):
        super(TotalVarianceLoss, self).__init__()

        self.num_channels = num_channels
        self.padding = (kernel_size - 1) // 2, (kernel_size - 1) // 2

        weights = np.ones([num_channels, 1, kernel_size, kernel_size], dtype=np.float32)
        weights /= kernel_size * kernel_size
        self.register_buffer('weights', torch.from_numpy(weights).cuda())

        self.hard_values = hard_values
        self.erase_threshold = threshold
        self.dilate_threshold = 3.0 / float(kernel_size * kernel_size)
        self.fraction = fraction
        self.limits = limits
        assert len(self.limits) == 2
        assert self.limits[0] < self.limits[1]

    def forward(self, values):
        with torch.no_grad():
            soft_values = F.conv2d(values, self.weights, None, 1, self.padding, 1, self.num_channels)

            if self.hard_values:
                erased_values = (soft_values < self.erase_threshold).float()
                dilated_values = F.conv2d(erased_values, self.weights, None, 1, self.padding, 1, self.num_channels)
                zeros_mask = dilated_values < self.dilate_threshold

                trg_values = torch.where(zeros_mask,
                                         torch.full_like(soft_values, self.limits[0]),
                                         torch.full_like(soft_values, self.limits[1]))
            else:
                trg_values = soft_values

        losses = torch.abs(values - trg_values)
        out = losses.mean()

        return out
