import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CircleLoss(nn.Module):
    def __init__(self, margin=0.2, scale=15.0):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, features, centers, labels, cam_ids):
        embeddings = F.normalize(features, p=2, dim=1)

        similarities = torch.mm(embeddings, torch.t(embeddings)).clamp(-1, 1)

        with torch.no_grad():
            pos_weights = F.relu((1.0 + self.margin) - similarities)
            neg_weights = F.relu(similarities + self.margin)

        pos_terms = self.scale * pos_weights * (similarities + (self.margin - 1.0))
        neg_terms = self.scale * neg_weights * (similarities - self.margin)

        with torch.no_grad():
            same_class_pairs = labels.view(-1, 1) == labels.view(1, -1)
            different_class_pairs = ~same_class_pairs

            ids_range = torch.arange(labels.size(0), device=labels.device)
            top_diagonal_pairs = ids_range.view(-1, 1) < ids_range.view(1, -1)

            pos_pairs = same_class_pairs & top_diagonal_pairs
            neg_pairs = different_class_pairs & top_diagonal_pairs

        pos_values = pos_terms[pos_pairs]
        neg_values = neg_terms[neg_pairs]
        all_pairs = neg_values.view(1, -1) - pos_values.view(-1, 1)

        loss = torch.log(1.0 + torch.exp(all_pairs).sum())

        return loss


class HardTripletLoss(nn.Module):
    def __init__(self, margin=0.35):
        super().__init__()
        self.margin = margin

    def forward(self, features, centers, labels, cam_ids):
        embeddings = F.normalize(features, p=2, dim=1)

        similarities = torch.mm(embeddings, torch.t(embeddings)).clamp(-1.0, 1.0)

        with torch.no_grad():
            same_class_pairs = labels.view(-1, 1) == labels.view(1, -1)
            different_class_pairs = ~same_class_pairs

            ids_range = torch.arange(labels.size(0), device=labels.device)
            non_diagonal_pairs = ids_range.view(-1, 1) != ids_range.view(1, -1)

            pos_pairs = same_class_pairs & non_diagonal_pairs
            neg_pairs = different_class_pairs

        s_pos, _ = torch.where(pos_pairs, similarities, torch.full_like(similarities, 1.0)).min(dim=1)
        s_neg, _ = torch.where(neg_pairs, similarities, torch.full_like(similarities, -1.0)).max(dim=1)

        # with torch.no_grad():
        #     w_pos = F.relu((1.0 + self.margin) - s_pos)
        #     w_neg = F.relu(s_neg + self.margin)
        #
        # term_pos = w_pos * (s_pos + (self.margin - 1.0))
        # term_neg = w_neg * (s_neg - self.margin)
        #
        # losses = F.softplus(term_neg - term_pos)
        # loss = losses.sum()

        losses = F.relu(self.margin + s_neg - s_pos)
        loss = losses.sum()

        num_valid = (losses > 0.0).sum().float()
        if num_valid > 0.0:
            loss /= num_valid

        return loss


class CenterLoss(nn.Module):
    """Implementation of the Center loss from https://ydwen.github.io/papers/WenECCV16.pdf"""

    def __init__(self, num_classes, embed_size):
        super().__init__()
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.randn(self.num_classes, embed_size).cuda())
        self.embed_size = embed_size

    def get_centers(self):
        """Returns estimated centers"""
        return self.centers

    def normalize_centers(self):
        self.centers.data = F.normalize(self.centers.data, p=2, dim=1)

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        batch_size = labels.size(0)
        features_dim = features.size(1)
        assert features_dim == self.embed_size

        centers = F.normalize(self.centers, p=2, dim=1)
        centers_batch = centers[labels, :]

        cos_diff = 1.0 - torch.sum(features * centers_batch, dim=1).clamp(-1, 1)
        center_loss = torch.sum(cos_diff) / batch_size

        return center_loss


class SampledCenterLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, centers, labels, cam_ids):
        centers = F.normalize(centers.detach(), p=2, dim=1)
        embeddings = F.normalize(features, p=2, dim=1)

        num_samples = 0
        loss = 0.0
        for class_id in labels:
            class_mask = labels == class_id

            class_embeddings = embeddings[class_mask]
            class_center = centers[class_id].view(1, -1)

            sampled_embeddings = self.sample_embeddings(class_embeddings)
            dist = 1.0 - torch.sum(sampled_embeddings * class_center, dim=1).clamp(-1, 1)

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

        with torch.no_grad():
            pair_dist = 1.0 - torch.sum(left_embeddings * right_embeddings, dim=1).clamp(-1, 1)
            threshold_dist = torch.median(pair_dist)
            pair_mask = pair_dist > threshold_dist

        left_embeddings = left_embeddings[pair_mask]
        right_embeddings = right_embeddings[pair_mask]

        ratio = torch.rand(left_embeddings.size(0), 1, dtype=embeddings.dtype, device=embeddings.device)
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

        pos_distances = 1.0 - torch.sum(features * centers_batch, dim=1).clamp(-1, 1)
        neg_distances = 1.0 - torch.mm(features, torch.t(centers)).clamp(-1, 1)
        losses = F.softplus(pos_distances.view(-1, 1) - neg_distances)

        pairs_valid_mask = different_class_pairs & (losses > 0.0)

        return losses, pairs_valid_mask


class SameCameraPushPlus(NormalizedLoss):
    def __init__(self):
        super().__init__()

    def _calculate(self, features, centers, labels, cam_ids):
        features = F.normalize(features, p=2, dim=1)

        centers = F.normalize(centers.detach(), p=2, dim=1)
        centers_batch = centers[labels, :]

        pos_distances = 1.0 - torch.sum(features * centers_batch, dim=1).clamp(-1, 1)
        neg_distances = 1.0 - torch.mm(features, torch.t(features)).clamp(-1, 1)
        losses = F.softplus(pos_distances.view(-1, 1) - neg_distances)

        different_class_pairs = labels.view(-1, 1) != labels.view(1, -1)
        same_camera_pairs = cam_ids.view(-1, 1) == cam_ids.view(1, -1)
        pairs_valid_mask = same_camera_pairs & different_class_pairs & (losses > 0.0)

        return losses, pairs_valid_mask


class CentersPush(NormalizedLoss):
    def __init__(self, margin=0.3):
        super().__init__()

        self.margin = margin

    def _calculate(self, features, centers, labels, cam_ids):
        centers = F.normalize(centers, p=2, dim=1)

        unique_labels = torch.unique(labels)
        unique_centers = centers[unique_labels, :]

        distances = 1.0 - torch.mm(unique_centers, torch.t(unique_centers)).clamp(-1, 1)
        losses = self.margin - distances

        different_class_pairs = unique_labels.view(-1, 1) != unique_labels.view(1, -1)
        pairs_valid_mask = different_class_pairs & (losses > 0.0)

        return losses, pairs_valid_mask


class SameIDPull(NormalizedLoss):
    def __init__(self):
        super().__init__()

    def _calculate(self, features, centers, labels, cam_ids):
        features = F.normalize(features, p=2, dim=1)

        losses = 1.0 - torch.mm(features, torch.t(features)).clamp(-1, 1)

        same_class_pairs = labels.view(-1, 1) == labels.view(1, -1)
        different_camera_pairs = cam_ids.view(-1, 1) != cam_ids.view(1, -1)
        pairs_valid_mask = different_camera_pairs & same_class_pairs

        return losses, pairs_valid_mask


class MetricLosses:
    """Class-aggregator for metric-learning losses"""

    def __init__(self, writer, num_classes, embed_size, center_coeff=1.0, glob_push_coeff=1.0,
                 local_push_coeff=1.0, pull_coeff=1.0, loss_balancing=True, centers_lr=0.5,
                 balancing_lr=0.01, name='ml'):
        self.writer = writer
        self.name = name

        self.total_losses_num = 0
        self.losses_map = dict()

        self.center_loss = CenterLoss(num_classes, embed_size)
        self.center_optimizer = torch.optim.SGD(self.center_loss.parameters(), lr=centers_lr)
        assert center_coeff >= 0
        self.center_coeff = center_coeff
        if self.center_coeff > 0:
            self.losses_map['center'] = self.total_losses_num
            self.total_losses_num += 1

        # self.sampled_center_loss = SampledCenterLoss()
        # if self.center_coeff > 0:
        #     self.losses_map['sampled_center'] = self.total_losses_num
        #     self.total_losses_num += 1

        # self.push_center_loss = CentersPush(margin=0.1)
        # if self.center_coeff > 0:
        #     self.losses_map['push_center'] = self.total_losses_num
        #     self.total_losses_num += 1

        self.glob_push_loss = HardTripletLoss(margin=0.35)
        assert glob_push_coeff >= 0
        self.glob_push_coeff = glob_push_coeff
        if self.glob_push_coeff > 0:
            self.losses_map['glob_push'] = self.total_losses_num
            self.total_losses_num += 1

        # self.local_push_loss = SameCameraPushPlus()
        # assert local_push_coeff >= 0
        # self.local_push_coeff = local_push_coeff
        # if self.local_push_coeff > 0:
        #     self.losses_map['local_push'] = self.total_losses_num
        #     self.total_losses_num += 1
        #
        # self.pull_loss = SameIDPull()
        # assert pull_coeff >= 0
        # self.pull_coeff = pull_coeff
        # if self.pull_coeff > 0:
        #     self.losses_map['pull'] = self.total_losses_num
        #     self.total_losses_num += 1

        self.loss_balancing = loss_balancing and self.total_losses_num > 1
        if self.loss_balancing:
            self.loss_weights = nn.Parameter(torch.FloatTensor(self.total_losses_num).cuda())
            self.balancing_optimizer = torch.optim.SGD([self.loss_weights], lr=balancing_lr)
            for i in range(self.total_losses_num):
                self.loss_weights.data[i] = 0.

    def _balance_losses(self, losses, scale=0.1):
        assert len(losses) == self.total_losses_num

        weighted_losses = []
        num_valid_losses = 0
        for i, loss_val in enumerate(losses):
            if loss_val > 0.0:
                weight = torch.exp(-self.loss_weights[i])
                weighted_loss_val = weight * loss_val + scale * self.loss_weights[i]

                losses[i] = weighted_loss_val.clamp_min(0.0)
                weighted_losses.append(weighted_loss_val)

                num_valid_losses += 1
            else:
                losses[i] = loss_val.clamp_min(0.0)
                weighted_losses.append(losses[i])

        scale = float(len(losses)) / float(num_valid_losses if num_valid_losses > 0 else 1)
        loss = scale * sum(losses)

        return loss, weighted_losses

    def __call__(self, features, glob_centers, labels, cam_ids, iteration):
        all_loss_values = []

        center_loss_val = 0
        # sampled_center_loss_val = 0
        # push_center_loss_val = 0
        if self.center_coeff > 0.:
            center_loss_val = self.center_loss(features, labels)
            all_loss_values.append(center_loss_val)

            # sampled_center_loss_val = self.sampled_center_loss(features, self.center_loss.get_centers(), labels, cam_ids)
            # all_loss_values.append(sampled_center_loss_val)

            # push_center_loss_val = self.push_center_loss(features, self.center_loss.get_centers(), labels, cam_ids)
            # all_loss_values.append(push_center_loss_val)

        glob_push_plus_loss_val = 0
        if self.glob_push_coeff > 0.0 and self.center_coeff > 0.0:
            glob_push_plus_loss_val = self.glob_push_loss(features, self.center_loss.get_centers(), labels, cam_ids)
            all_loss_values.append(glob_push_plus_loss_val)

        # local_push_loss_val = 0
        # if self.local_push_coeff > 0.0 and self.center_coeff > 0.0:
        #     local_push_loss_val = self.local_push_loss(features, self.center_loss.get_centers(), labels, cam_ids)
        #     all_loss_values.append(local_push_loss_val)
        #
        # pull_loss_val = 0
        # if self.pull_coeff > 0.0 and self.center_coeff > 0.0:
        #     pull_loss_val = self.pull_loss(features, self.center_loss.get_centers(), labels, cam_ids)
        #     all_loss_values.append(pull_loss_val)

        if self.loss_balancing and self.total_losses_num > 1:
            loss_value, weighted_loss_values = self._balance_losses(all_loss_values)
        else:
            loss_value = self.center_coeff * center_loss_val + \
                         self.glob_push_coeff * glob_push_plus_loss_val
            weighted_loss_values = [0.0] * self.total_losses_num
        self.last_loss_value = loss_value

        if self.writer is not None:
            if self.center_coeff > 0.:
                self.writer.add_scalar(
                    'Loss/{}/center'.format(self.name), center_loss_val, iteration)
                if self.loss_balancing:
                    self.writer.add_scalar(
                        'Aux/{}/center_w'.format(self.name),
                        weighted_loss_values[self.losses_map['center']],
                        iteration)

                # self.writer.add_scalar(
                #     'Loss/{}/sampled_center'.format(self.name), sampled_center_loss_val, iteration)
                # if self.loss_balancing:
                #     self.writer.add_scalar(
                #         'Aux/{}/sampled_center_w'.format(self.name),
                #         weighted_loss_values[self.losses_map['sampled_center']],
                #         iteration)

                # self.writer.add_scalar(
                #     'Loss/{}/push_center'.format(self.name), push_center_loss_val,
                #     iteration)
                # if self.loss_balancing:
                #     self.writer.add_scalar(
                #         'Aux/{}/push_center_w'.format(self.name),
                #         weighted_loss_values[self.losses_map['push_center']],
                #         iteration)

                if self.glob_push_coeff > 0.0:
                    self.writer.add_scalar(
                        'Loss/{}/global_push'.format(self.name), glob_push_plus_loss_val,
                        iteration)
                    if self.loss_balancing:
                        self.writer.add_scalar(
                            'Aux/{}/global_push_w'.format(self.name),
                            weighted_loss_values[self.losses_map['glob_push']],
                            iteration)

                # if self.local_push_coeff > 0.0:
                #     self.writer.add_scalar(
                #         'Loss/{}/local_push'.format(self.name), local_push_loss_val,
                #         iteration)
                #     if self.loss_balancing:
                #         self.writer.add_scalar(
                #             'Aux/{}/local_push_w'.format(self.name),
                #             weighted_loss_values[self.losses_map['local_push']],
                #             iteration)
                #
                # if self.pull_coeff > 0.0:
                #     self.writer.add_scalar(
                #         'Loss/{}/pull'.format(self.name), pull_loss_val,
                #         iteration)
                #     if self.loss_balancing:
                #         self.writer.add_scalar(
                #             'Aux/{}/pull_w'.format(self.name),
                #             weighted_loss_values[self.losses_map['pull']],
                #             iteration)

            if self.total_losses_num > 0:
                self.writer.add_scalar('Loss/{}/AUX_losses'.format(self.name), loss_value, iteration)

        return loss_value

    def init_iteration(self):
        """Initializes a training iteration"""

        if self.center_coeff > 0.:
            self.center_optimizer.zero_grad()

        if self.loss_balancing:
            self.balancing_optimizer.zero_grad()

    def end_iteration(self):
        """Finalizes a training iteration"""

        self.last_loss_value.backward(retain_graph=True)

        if self.center_coeff > 0.:
            self.center_optimizer.step()
            self.center_loss.normalize_centers()

        if self.loss_balancing:
            self.balancing_optimizer.step()
