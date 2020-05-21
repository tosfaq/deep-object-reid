"""
 Copyright (c) 2019 Intel Corporation

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

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvRegularizer(nn.Module):
    def __init__(self, reg_class, **kwargs):
        super().__init__()

        self.reg_instance = reg_class(**kwargs)

    def forward(self, net):
        num_losses = 0
        accumulator = torch.tensor(0.0).cuda()
        for module in net.module.modules():
            if not isinstance(module, nn.Conv2d):
                continue

            loss = self.reg_instance(module.weight)
            if loss > 0:
                accumulator += loss
                num_losses += 1

        return accumulator / float(max(1, num_losses))


class SVMORegularizer(nn.Module):
    def __init__(self, beta):
        super().__init__()

        self.beta = beta

    @staticmethod
    def dominant_eigenvalue(A):  # A: 'N x N'
        x = torch.rand(A.size(0), 1, dtype=A.dtype, device=A.device)

        Ax = (A @ x)
        AAx = (A @ Ax)
        value = AAx.permute(1, 0) @ Ax / (Ax.permute(1, 0) @ Ax)

        return value

    def get_singular_values(self, A):  # A: 'M x N, M >= N'
        ATA = A.permute(1, 0) @ A
        N, _ = ATA.size()

        largest = self.dominant_eigenvalue(ATA)

        I = largest * torch.eye(N, dtype=A.dtype, device=A.device)
        smallest = self.dominant_eigenvalue(ATA - I)

        return largest, smallest

    def forward(self, W):  # W: 'S x C x H x W'
        old_size = W.size()
        if old_size[0] == 1:
            return 0

        W = W.view(old_size[0], -1).permute(1, 0)  # (C x H x W) x S
        largest, smallest = self.get_singular_values(W)

        loss = self.beta * (largest - smallest) ** 2

        return loss.squeeze()


class HardDecorrelationRegularizer(nn.Module):
    def __init__(self, max_score, scale):
        super().__init__()

        self.max_score = float(max_score)
        assert -1.0 < self.max_score < 1.0
        self.scale = float(scale)
        assert self.scale > 0.0

    def forward(self, W):
        num_filters = W.size(0)
        if num_filters == 1:
            return 0

        W = W.view(num_filters, -1)

        dim = W.size(1)
        if dim < num_filters:
            return 0

        W = F.normalize(W, p=2, dim=-1)
        similarities = torch.matmul(W, W.t())

        losses = torch.triu(similarities.abs().clamp_min(self.max_score) ** 2, diagonal=1)
        filtered_losses = losses[losses > 0.0]
        loss = filtered_losses.mean() if filtered_losses.numel() > 0 else filtered_losses.sum()

        return self.scale * loss


class NormRegularizer(nn.Module):
    def __init__(self, scale):
        super().__init__()

        self.scale = scale

    def forward(self, W):
        num_filters = W.size(0)
        if num_filters == 1:
            return 0

        W = W.view(num_filters, -1)
        norms = torch.sqrt(torch.sum(W ** 2, dim=-1))

        trg_norm = torch.median(norms.detach())
        losses = (norms - trg_norm) ** 2
        loss = losses.mean()

        return self.scale * loss


class ComposeRegularizer(nn.Module):
    def __init__(self, regularizers):
        super().__init__()

        self.regularizers = regularizers
        assert len(regularizers) > 0

    def forward(self, net):
        loss = 0
        for regularizer in self.regularizers:
            loss += regularizer(net)

        return loss


def get_regularizer(cfg_reg):
    regularizers = []

    if cfg_reg.ow:
        regularizers.append(ConvRegularizer(
            SVMORegularizer,
            beta=cfg_reg.ow_beta
        ))

    if cfg_reg.nw:
        regularizers.append(ConvRegularizer(
            NormRegularizer,
            scale=cfg_reg.nw_scale
        ))

    if cfg_reg.hd:
        regularizers.append(ConvRegularizer(
            HardDecorrelationRegularizer,
            max_score=cfg_reg.hd_max_score,
            scale=cfg_reg.hd_scale
        ))

    if len(regularizers) > 0:
        return ComposeRegularizer(regularizers)
    else:
        return None
