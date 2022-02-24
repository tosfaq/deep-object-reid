"""
 Copyright (c) 2020-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter


class AngleSimpleLinear(nn.Module):
    """Computes cos of angles between input vectors and weights vectors"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # create proxy weights
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.normal_().renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        cos_theta = F.normalize(x.view(x.shape[0], -1), dim=1).mm(F.normalize(self.weight, p=2, dim=0))
        return cos_theta.clamp(-1, 1)

    def get_centers(self):
        return torch.t(self.weight)


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class AMSoftmaxLoss(nn.Module):
    margin_types = ['cos', 'arc']

    def __init__(self, use_gpu=True, margin_type='cos', gamma=0.0, m=0.5,
                 s=30, conf_penalty=0.0, label_smooth=0, aug_type='', pr_product=False,
                 symmetric_ce=False):
        super().__init__()
        self.use_gpu = use_gpu
        self.conf_penalty = conf_penalty
        self.label_smooth = label_smooth
        self.aug_type = aug_type
        self.pr_product = pr_product

        assert margin_type in AMSoftmaxLoss.margin_types
        self.margin_type = margin_type
        assert gamma >= 0
        self.gamma = gamma
        assert m >= 0
        self.m = m
        assert s > 0
        self.scale = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.symmetric_ce = symmetric_ce
        self.class_margins = self.m

    @staticmethod
    def _valid(params):
        if isinstance(params, (list, tuple)):
            for p in params:
                if p is None:
                    return False
        else:
            if params is None:
                return False
        return True

    def forward(self, cos_theta, target, aug_index=None, lam=None, scale=None):
        """
        Args:
            cos_theta (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            target (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
            iteration (int): current iteration
        """
        logits_aug_avai = self._valid([self.aug_type, aug_index, lam]) # augmentations like fmix, cutmix, augmix
        self.scale = scale if scale else self.scale # different scale for different models
        if logits_aug_avai:
            targets1 = torch.zeros(cos_theta.size(), device=target.device).scatter_(1, target.detach().unsqueeze(1), 1)
            targets2 = targets1[aug_index]
            new_targets = targets1 * lam + targets2 * (1 - lam)
            # in case if target label changed
            target = new_targets.argmax(dim=1)

        if self.pr_product:
            pr_alpha = torch.sqrt(1.0 - cos_theta.pow(2.0))
            cos_theta = pr_alpha.detach() * cos_theta + cos_theta.detach() * (1.0 - pr_alpha)

        one_hot_target = torch.zeros_like(cos_theta, dtype=torch.uint8)
        one_hot_target.scatter_(1, target.data.view(-1, 1), 1)
        # change margins accordingly
        self.class_margins *= one_hot_target

        if self.margin_type == 'cos':
            phi_theta = cos_theta - self.class_margins
        else:
            self.cos_m *= one_hot_target
            self.sin_m *= one_hot_target
            sine = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
            phi_theta = cos_theta * self.cos_m - sine * self.sin_m
            phi_theta = torch.where(cos_theta > self.th, phi_theta, cos_theta - self.sin_m * self.class_margins)

        output = phi_theta

        if self.gamma == 0.0:
            output *= self.scale

            if self.label_smooth > 0:
                assert not self.aug_type
                target = torch.zeros(output.size(), device=target.device).scatter_(1, target.detach().unsqueeze(1), 1)
                num_classes = output.size(1)
                target = (1.0 - self.label_smooth) * target + self.label_smooth / float(num_classes)
                losses = (- target * F.log_softmax(output, dim=1)).sum(dim=1)

            elif logits_aug_avai:
                losses = (- new_targets * F.log_softmax(output, dim=1)).sum(dim=1)

            else:
                losses = F.cross_entropy(output, target, reduction='none')

            if self.symmetric_ce:
                all_probs = F.softmax(output, dim=-1)
                target_probs = all_probs[torch.arange(target.size(0), device=target.device), target]
                losses += 4.0 * (1.0 - target_probs)

            if self.conf_penalty > 0.0:
                probs = F.softmax(output, dim=1)
                log_probs = F.log_softmax(output, dim=1)
                entropy = torch.sum(-probs * log_probs, dim=1)

                losses = F.relu(losses - self.conf_penalty * entropy)

            with torch.no_grad():
                nonzero_count = max(losses.nonzero().size(0), 1)

            return losses.sum() / nonzero_count

        return focal_loss(F.cross_entropy(self.scale * output, target, reduction='none'), self.gamma)
