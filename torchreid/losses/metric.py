import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    """Implementation of the Center loss from https://ydwen.github.io/papers/WenECCV16.pdf"""

    def __init__(self, num_classes, embed_size, cos_dist=True):
        super().__init__()
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.randn(self.num_classes, embed_size).cuda())
        self.embed_size = embed_size

        self.cos_dist = cos_dist
        if self.cos_dist:
            self.cos_sim = nn.CosineSimilarity()
        else:
            self.mse = nn.MSELoss(reduction='elementwise_mean')

    def get_centers(self):
        """Returns estimated centers"""
        return self.centers

    def forward(self, features, labels):
        features = F.normalize(features)
        batch_size = labels.size(0)
        features_dim = features.size(1)
        assert features_dim == self.embed_size

        if self.cos_dist:
            self.centers.data = F.normalize(self.centers.data, p=2, dim=1)

        centers_batch = self.centers[labels, :]

        if self.cos_dist:
            cos_diff = 1.0 - self.cos_sim(features, centers_batch)
            center_loss = torch.sum(cos_diff) / batch_size
        else:
            center_loss = self.mse(centers_batch, features)

        return center_loss


class SampledCenterLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, centers, labels, cam_ids):
        embeddings = F.normalize(features, p=2, dim=1)
        centers = F.normalize(centers.detach(), p=2, dim=1)

        num_samples = 0
        loss = 0.0
        for class_id in labels:
            class_mask = labels == class_id

            class_embeddings = embeddings[class_mask]
            class_center = centers[class_id].view(1, -1)

            sampled_embeddings = self.sample_embeddings(class_embeddings)
            dist = 1.0 - torch.sum(sampled_embeddings * class_center, dim=1)

            loss += dist.sum()
            num_samples += dist.numel()

        out_loss = loss / num_samples if num_samples > 0 else 0.0

        return out_loss

    @staticmethod
    def sample_embeddings(embeddings):
        with torch.no_grad():
            n = embeddings.size(0)
            left_ids = torch.from_numpy(np.array([i for i in range(n) for j in range(i + 1, n)])).cuda()
            right_ids = torch.from_numpy(np.array([j for i in range(n) for j in range(i + 1, n)])).cuda()

        left_embeddings = torch.index_select(embeddings, 0, left_ids)
        right_embeddings = torch.index_select(embeddings, 0, right_ids)

        # with torch.no_grad():
        #     cos_gamma = torch.sum(left_embeddings * right_embeddings, dim=1).clamp(-1.0, 1.0)
        #
        #     valid_mask = cos_gamma > 0.1
        #     cos_gamma = cos_gamma[valid_mask]
        #     sin_gamma = torch.sqrt((1.0 - cos_gamma ** 2).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        #
        #     ratio = torch.rand_like(cos_gamma)
        #     alpha_angle =\
        #         torch.acos((1.0 - ratio) / torch.sqrt((ratio * ratio - 2.0 * ratio * cos_gamma + 1.0).clamp(0.0, 1.0))) + \
        #         torch.atan(ratio * sin_gamma / (ratio * cos_gamma - 1.0))
        #     gamma_angle = torch.acos(cos_gamma)
        #     betta_angle = gamma_angle - alpha_angle
        #
        #     left_scale = (torch.sin(betta_angle) / sin_gamma).view(-1, 1)
        #     right_scale = (torch.sin(alpha_angle) / sin_gamma).view(-1, 1)
        #
        # sampled_embeddings = left_scale * left_embeddings[valid_mask] + right_scale * right_embeddings[valid_mask]

        ratio = torch.rand(left_ids.size(0), 1, dtype=embeddings.dtype, device=embeddings.device)
        sampled_embeddings = F.normalize(ratio * left_embeddings + (1.0 - ratio) * right_embeddings, p=2, dim=1)

        return sampled_embeddings


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
                 local_push_coeff=1.0, pull_coeff=1.0, loss_balancing=True, centers_lr=0.5,
                 balancing_lr=0.01, name='ml'):
        self.writer = writer
        self.total_losses_num = 0
        self.name = name

        self.center_loss = CenterLoss(num_classes, embed_size, cos_dist=True)
        self.optimizer_centloss = torch.optim.SGD(self.center_loss.parameters(), lr=centers_lr)
        assert center_coeff >= 0
        self.center_coeff = center_coeff
        if self.center_coeff > 0:
            self.total_losses_num += 1

        self.glob_push_loss = GlobalPushPlus()
        assert glob_push_coeff >= 0
        self.glob_push_coeff = glob_push_coeff
        if self.glob_push_coeff > 0:
            self.total_losses_num += 1

        self.local_push_loss = SameCameraPushPlus()
        assert local_push_coeff >= 0
        self.local_push_coeff = local_push_coeff
        if self.local_push_coeff > 0:
            self.total_losses_num += 1

        self.pull_loss = SameIDPull()
        assert pull_coeff >= 0
        self.pull_coeff = pull_coeff
        if self.pull_coeff > 0:
            self.total_losses_num += 1

        self.sampled_center_loss = SampledCenterLoss()
        if self.center_coeff > 0:
            self.total_losses_num += 1

        self.loss_balancing = loss_balancing and self.total_losses_num > 1
        if self.loss_balancing:
            self.loss_weights = nn.Parameter(torch.FloatTensor(self.total_losses_num).cuda())
            self.balancing_optimizer = torch.optim.SGD([self.loss_weights], lr=balancing_lr)
            for i in range(self.total_losses_num):
                self.loss_weights.data[i] = 0.

    def _balance_losses(self, losses, scale=0.1):
        assert len(losses) == self.total_losses_num

        weighted_losses = []
        for i, loss_val in enumerate(losses):
            weight = torch.exp(-self.loss_weights[i])
            weighted_loss_val = weight * loss_val + scale * self.loss_weights[i]

            losses[i] = weighted_loss_val.clamp_min(0.0)
            weighted_losses.append(weighted_loss_val)

        return sum(losses), weighted_losses

    def __call__(self, features, glob_centers, labels, cam_ids, iteration):
        all_loss_values = []

        center_loss_val = 0
        sampled_center_loss_val = 0
        if self.center_coeff > 0.:
            center_loss_val = self.center_loss(features, labels)
            all_loss_values.append(center_loss_val)
            self.last_center_val = center_loss_val

            sampled_center_loss_val = self.sampled_center_loss(features, self.center_loss.get_centers(), labels, cam_ids)
            all_loss_values.append(sampled_center_loss_val)

        glob_push_plus_loss_val = 0
        if self.glob_push_coeff > 0.0 and self.center_coeff > 0.0:
            glob_push_plus_loss_val = self.glob_push_loss(features, self.center_loss.get_centers(), labels, cam_ids)
            all_loss_values.append(glob_push_plus_loss_val)

        local_push_loss_val = 0
        if self.local_push_coeff > 0.0 and self.center_coeff > 0.0:
            local_push_loss_val = self.local_push_loss(features, self.center_loss.get_centers(), labels, cam_ids)
            all_loss_values.append(local_push_loss_val)

        pull_loss_val = 0
        if self.pull_coeff > 0.0 and self.center_coeff > 0.0:
            pull_loss_val = self.pull_loss(features, self.center_loss.get_centers(), labels, cam_ids)
            all_loss_values.append(pull_loss_val)

        if self.loss_balancing and self.total_losses_num > 1:
            loss_value, weighted_loss_values = self._balance_losses(all_loss_values)
            loss_value *= self.center_coeff
            self.last_loss_value = loss_value
        else:
            loss_value = self.center_coeff * (center_loss_val + sampled_center_loss_val) + \
                         self.glob_push_coeff * glob_push_plus_loss_val + \
                         self.local_push_coeff * local_push_loss_val + \
                         self.pull_coeff * pull_loss_val
            weighted_loss_values = [0.0] * self.total_losses_num

        if self.writer is not None:
            if self.center_coeff > 0.:
                self.writer.add_scalar(
                    'Loss/{}/center'.format(self.name), center_loss_val, iteration)
                if self.loss_balancing:
                    self.writer.add_scalar(
                        'Aux/{}/center_w'.format(self.name), weighted_loss_values[0], iteration)

                self.writer.add_scalar(
                    'Loss/{}/sampled_center'.format(self.name), sampled_center_loss_val, iteration)
                if self.loss_balancing:
                    self.writer.add_scalar(
                        'Aux/{}/sampled_center_w'.format(self.name), weighted_loss_values[1], iteration)

                if self.glob_push_coeff > 0.0:
                    self.writer.add_scalar(
                        'Loss/{}/global_push'.format(self.name), glob_push_plus_loss_val, iteration)
                    if self.loss_balancing:
                        self.writer.add_scalar(
                            'Aux/{}/global_push_w'.format(self.name), weighted_loss_values[2], iteration)

                if self.local_push_coeff > 0.0:
                    self.writer.add_scalar(
                        'Loss/{}/local_push'.format(self.name), local_push_loss_val, iteration)
                    if self.loss_balancing:
                        self.writer.add_scalar(
                            'Aux/{}/local_push_w'.format(self.name), weighted_loss_values[3], iteration)

                if self.pull_coeff > 0.0:
                    self.writer.add_scalar(
                        'Loss/{}/pull'.format(self.name), pull_loss_val, iteration)
                    if self.loss_balancing:
                        self.writer.add_scalar(
                            'Aux/{}/pull_w'.format(self.name), weighted_loss_values[4], iteration)

            if self.total_losses_num > 0:
                self.writer.add_scalar(
                    'Loss/{}/AUX_losses'.format(self.name), loss_value, iteration)

        return loss_value

    def init_iteration(self):
        """Initializes a training iteration"""

        if self.center_coeff > 0.:
            self.optimizer_centloss.zero_grad()

        if self.loss_balancing:
            self.balancing_optimizer.zero_grad()

    def end_iteration(self):
        """Finalizes a training iteration"""

        if self.loss_balancing and self.total_losses_num > 1:
            self.last_loss_value.backward(retain_graph=True)
            self.balancing_optimizer.step()

        if self.center_coeff > 0.:
            self.last_center_val.backward(retain_graph=True)
            for param in self.center_loss.parameters():
                param.grad.data *= (1. / self.center_coeff)
            self.optimizer_centloss.step()
