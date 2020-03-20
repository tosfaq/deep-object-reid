import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CenterLoss(nn.Module):
    """Implementation of the Center loss from https://ydwen.github.io/papers/WenECCV16.pdf"""

    def __init__(self):
        super().__init__()

    def forward(self, features, centers, labels):
        features = F.normalize(features, p=2, dim=1)

        centers = F.normalize(centers.detach(), p=2, dim=1)
        centers_batch = centers[labels, :]

        center_distances = 1.0 - torch.sum(features * centers_batch, dim=1)
        loss = center_distances.mean()

        return loss


class NormalizedLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, centers, labels, cam_ids):
        losses, pairs_valid_mask = self._calculate(features, centers, labels, cam_ids)
        losses = torch.where(pairs_valid_mask, losses, torch.zeros_like(losses))

        num_valid = pairs_valid_mask.sum().float()
        loss = losses.sum()
        if num_valid > 0.0:
            loss /= num_valid

        # num_valid_pairs = pairs_valid_mask.sum(dim=1)
        # num_valid_pairs = torch.where(num_valid_pairs > 0, num_valid_pairs, torch.ones_like(num_valid_pairs))
        # sample_losses = losses.sum(dim=1) / num_valid_pairs.float()
        #
        # samples_valid_mask = sample_losses > 0.0
        # num_valid_samples = torch.sum(samples_valid_mask)
        # loss = sample_losses.sum()
        # if num_valid_samples > 0:
        #     loss /= num_valid_samples

        return loss

    def _calculate(self, features, centers, labels, cam_ids):
        raise NotImplementedError


class GlobalPushPlus(NormalizedLoss):
    """Implementation of the Global Push Plus loss from https://arxiv.org/abs/1812.02465"""

    def __init__(self):
        super().__init__()

    def _calculate(self, features, centers, labels, cam_ids):
        features = F.normalize(features, p=2, dim=1)

        centers = F.normalize(centers.detach(), p=2, dim=1)
        centers_batch = centers[labels, :]

        num_classes = centers.shape[0]
        center_ids = torch.arange(num_classes, dtype=labels.dtype, device=labels.device)
        different_class_pairs = labels.view(-1, 1) != center_ids.view(1, -1)

        pos_distances = 1.0 - torch.sum(features * centers_batch, dim=1)
        neg_distances = 1.0 - torch.mm(features, torch.t(centers))
        losses = F.softplus(pos_distances.view(-1, 1) - neg_distances)

        pairs_valid_mask = different_class_pairs * (losses > 0.0)

        return losses, pairs_valid_mask


class SameCameraPushPlus(NormalizedLoss):
    def __init__(self):
        super().__init__()

    def _calculate(self, features, centers, labels, cam_ids):
        features = F.normalize(features, p=2, dim=1)

        centers = F.normalize(centers.detach(), p=2, dim=1)
        centers_batch = centers[labels, :]

        pos_distances = 1.0 - torch.sum(features * centers_batch, dim=1)
        neg_distances = 1.0 - torch.mm(features, torch.t(features))
        losses = F.softplus(pos_distances.view(-1, 1) - neg_distances)

        different_class_pairs = labels.view(-1, 1) != labels.view(1, -1)
        same_camera_pairs = cam_ids.view(-1, 1) == cam_ids.view(1, -1)
        pairs_valid_mask = same_camera_pairs * different_class_pairs * (losses > 0.0)

        return losses, pairs_valid_mask


class SameIDPull(NormalizedLoss):
    def __init__(self):
        super().__init__()

    def _calculate(self, features, centers, labels, cam_ids):
        features = F.normalize(features, p=2, dim=1)

        losses = 1.0 - torch.mm(features, torch.t(features))

        same_class_pairs = labels.view(-1, 1) == labels.view(1, -1)
        different_camera_pairs = cam_ids.view(-1, 1) != cam_ids.view(1, -1)
        pairs_valid_mask = different_camera_pairs * same_class_pairs

        return losses, pairs_valid_mask


class MetricLosses:
    """Class-aggregator for metric-learning losses"""

    def __init__(self, writer, num_classes, embed_size, center_coeff=1.0, glob_push_coeff=1.0,
                 local_push_coeff=1.0, pull_coeff=1.0, track_centers=False, new_center_weight=0.5,
                 name='ml'):
        self.writer = writer
        self.center_coeff = center_coeff
        self.glob_push_coeff = glob_push_coeff
        self.local_push_coeff = local_push_coeff
        self.pull_coeff = pull_coeff
        self.name = name

        self.center_loss = CenterLoss()
        self.glob_push_loss = GlobalPushPlus()
        self.local_push_loss = SameCameraPushPlus()
        self.pull_loss = SameIDPull()

        self.track_centers = track_centers
        self.new_center_weight = new_center_weight
        if self.track_centers:
            self.centers = np.random.normal(size=[num_classes, embed_size]).astype(np.float32)

    def __call__(self, features, glob_centers, labels, cam_ids, iteration):
        if self.track_centers:
            centers = torch.from_numpy(self.centers).cuda()
        else:
            centers = glob_centers

        center_loss_val = self.center_loss(features, centers, labels)
        if self.writer is not None:
            self.writer.add_scalar('Loss/{}/center'.format(self.name), center_loss_val, iteration)

        glob_push_loss_val = self.glob_push_loss(features, centers, labels, cam_ids)
        if self.writer is not None:
            self.writer.add_scalar('Loss/{}/global_push'.format(self.name), glob_push_loss_val, iteration)

        # local_push_loss_val = self.local_push_loss(features, centers, labels, cam_ids)
        # if self.writer is not None:
        #     self.writer.add_scalar('Loss/{}/local_push'.format(self.name), local_push_loss_val, iteration)
        #
        # pull_loss_val = self.pull_loss(features, centers, labels, cam_ids)
        # if self.writer is not None:
        #     self.writer.add_scalar('Loss/{}/pull'.format(self.name), pull_loss_val, iteration)

        # total_loss = self.center_coeff * center_loss_val +\
        #              self.glob_push_coeff * glob_push_loss_val +\
        #              self.local_push_coeff * local_push_loss_val +\
        #              self.pull_coeff * pull_loss_val
        total_loss = self.center_coeff * center_loss_val + \
                     self.glob_push_coeff * glob_push_loss_val
        if self.writer is not None:
            self.writer.add_scalar('Loss/{}/AUX_losses'.format(self.name), total_loss, iteration)

        if self.track_centers:
            with torch.no_grad():
                for class_id in labels:
                    class_mask = labels == class_id

                    class_features = features[class_mask]
                    class_embedding = F.normalize(class_features.mean(dim=0), p=2, dim=0)

                    center = F.normalize(centers[class_id], p=2, dim=0)
                    new_center = (1.0 - self.new_center_weight) * center + \
                                 self.new_center_weight * class_embedding

                    centers[class_id, :] = F.normalize(new_center, p=2, dim=0)

                self.centers = centers.data.cpu().numpy()

        return total_loss

    def init_iteration(self):
        pass

    def end_iteration(self):
        pass
