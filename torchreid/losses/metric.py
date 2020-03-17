import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    """Implementation of the Center loss from https://ydwen.github.io/papers/WenECCV16.pdf"""

    def __init__(self):
        super().__init__()

    def forward(self, features, centers, labels):
        features = F.normalize(features, p=2, dim=1)

        centers = F.normalize(centers, p=2, dim=1)
        centers_batch = centers[labels, :]

        center_distances = 1.0 - torch.sum(features * centers_batch, dim=1)
        loss = center_distances.mean()

        return loss


class GlobalPushPlus(nn.Module):
    """Implementation of the Global Push Plus loss from https://arxiv.org/abs/1812.02465"""

    def __init__(self):
        super().__init__()

    def forward(self, features, centers, labels):
        features = F.normalize(features, p=2, dim=1)

        centers = F.normalize(centers, p=2, dim=1)
        centers_batch = centers[labels, :]

        num_classes = centers.shape[0]
        center_ids = torch.arange(num_classes, dtype=labels.dtype, device=labels.device)
        different_class_pairs = labels.view(-1, 1) != center_ids.view(1, -1)

        pos_distances = 1.0 - torch.sum(features * centers_batch, dim=1)
        neg_distances = 1.0 - torch.mm(features, torch.t(centers))
        losses = pos_distances.view(-1, 1) - neg_distances

        pairs_valid_mask = different_class_pairs * (losses > 0.0)
        num_valid_pairs = pairs_valid_mask.sum(dim=1)
        num_valid_pairs = torch.where(num_valid_pairs > 0, num_valid_pairs, torch.ones_like(num_valid_pairs))
        sample_losses = (losses * pairs_valid_mask).sum(dim=1) / num_valid_pairs.float()

        samples_valid_mask = sample_losses > 0.0
        num_valid_samples = torch.sum(samples_valid_mask)
        loss = sample_losses.sum()
        if num_valid_samples > 0:
            loss /= num_valid_samples

        return loss


class LocalPushPlus(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, centers, labels):
        features = F.normalize(features, p=2, dim=1)

        centers = F.normalize(centers, p=2, dim=1)
        centers_batch = centers[labels, :]

        pos_distances = 1.0 - torch.sum(features * centers_batch, dim=1)
        neg_distances = 1.0 - torch.mm(features, torch.t(features))
        losses = pos_distances.view(-1, 1) - neg_distances

        different_class_pairs = labels.view(-1, 1) != labels.view(1, -1)
        pairs_valid_mask = different_class_pairs * (losses > 0.0)
        losses = torch.where(pairs_valid_mask, losses, torch.zeros_like(losses))

        num_valid_pairs = pairs_valid_mask.sum(dim=1)
        num_valid_pairs = torch.where(num_valid_pairs > 0, num_valid_pairs, torch.ones_like(num_valid_pairs))
        sample_losses = losses.sum(dim=1) / num_valid_pairs.float()

        samples_valid_mask = sample_losses > 0.0
        num_valid_samples = torch.sum(samples_valid_mask)
        loss = sample_losses.sum()
        if num_valid_samples > 0:
            loss /= num_valid_samples

        return loss


class MetricLosses:
    """Class-aggregator for metric-learning losses"""

    def __init__(self, writer, center_coeff=1.0, glob_push_coeff=1.0):
        self.writer = writer
        self.center_coeff = center_coeff
        self.glob_push_coeff = 0.5 * glob_push_coeff

        self.center_loss = CenterLoss()
        self.glob_push_loss = GlobalPushPlus()
        self.local_push_loss = LocalPushPlus()

    def __call__(self, features, centers, labels, iteration, name='ml'):
        center_loss_val = self.center_loss(features, centers, labels)
        if self.writer is not None:
            self.writer.add_scalar('Loss/{}/center'.format(name), center_loss_val, iteration)

        glob_push_loss_val = self.glob_push_loss(features, centers, labels)
        if self.writer is not None:
            self.writer.add_scalar('Loss/{}/global_push'.format(name), glob_push_loss_val, iteration)

        local_push_loss_val = self.local_push_loss(features, centers, labels)
        if self.writer is not None:
            self.writer.add_scalar('Loss/{}/local_push'.format(name), glob_push_loss_val, iteration)

        total_loss = self.center_coeff * center_loss_val +\
                     self.glob_push_coeff * (glob_push_loss_val + local_push_loss_val)

        if self.writer is not None:
            self.writer.add_scalar('Loss/{}/AUX_losses'.format(name), total_loss, iteration)

        return total_loss
