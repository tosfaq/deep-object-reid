import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TotalVarianceLoss(nn.Module):
    def __init__(self, kernel_size, num_channels, hard_values=True, limits=(0.0, 1.0), threshold=0.5, fraction=0.1):
        super(TotalVarianceLoss, self).__init__()

        self.num_channels = num_channels
        # self.padding = (kernel_size - 1) // 2, (kernel_size - 1) // 2

        weights = np.ones([num_channels, 1, kernel_size, kernel_size], dtype=np.float32)
        weights /= kernel_size * kernel_size
        self.register_buffer('weights', torch.from_numpy(weights).cuda())

        self.hard_values = hard_values
        self.threshold = threshold
        self.fraction = fraction
        self.limits = limits
        assert len(self.limits) == 2
        assert self.limits[0] < self.limits[1]

    def forward(self, values):
        with torch.no_grad():
            soft_values = F.conv2d(values, self.weights, None, 1, 0, 1, self.num_channels)
            n, c, h, w = values.size()

            if self.hard_values:
                num_samples = max(1, int(self.fraction * h * w))
                sorted_values, _ = torch.sort(soft_values.view(n, c, -1), dim=-1, descending=False)
                # zero_threshold = sorted_values[:, :, num_samples].view(n, c, 1, 1)
                # one_threshold = sorted_values[:, :, -num_samples].view(n, c, 1, 1)
                threshold = sorted_values[:, :, int(0.5 * float(h * w))].view(n, c, 1, 1)

                # zeros_mask = soft_values <= zero_threshold
                # none_ones_mask = soft_values < one_threshold
                # adaptive_mask = soft_values < self.threshold
                # final_zeros_mask = (zeros_mask | adaptive_mask) & none_ones_mask
                final_zeros_mask = soft_values <= threshold

                trg_values = torch.where(final_zeros_mask,
                                         torch.full_like(soft_values, self.limits[0]),
                                         torch.full_like(soft_values, self.limits[1]))
            else:
                trg_values = soft_values

        losses = torch.abs(values[:, :, 1:-1, 1:-1] - trg_values)
        out = losses.mean()

        return out
