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

            accumulator += self.reg_instance(module.weight)
            num_losses += 1

        return accumulator / float(max(1.0, num_losses))


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


class NormRegularizer(nn.Module):
    def __init__(self, max_factor, scale):
        super().__init__()

        self.max_factor = max_factor
        self.scale = scale

    def forward(self, W):
        num_filters = W.size(0)
        if num_filters == 1:
            return 0

        W = W.view(num_filters, -1)
        norms = torch.sqrt(torch.sum(W ** 2, dim=-1))

        with torch.no_grad():
            max_norm = torch.max(norms)
            mask = max_norm / norms > float(self.max_factor)
            num_invalid = mask.float().sum()

            trg_norm = torch.median(norms)

        losses = (norms - trg_norm) ** 2
        loss = losses[mask].sum()
        if num_invalid > 0.0:
            loss /= num_invalid

        return self.scale * loss


def get_regularizer(cfg_reg):
    if cfg_reg.ow:
        return ConvRegularizer(SVMORegularizer,
                               beta=cfg_reg.ow_beta)
    elif cfg_reg.nw:
        return ConvRegularizer(NormRegularizer,
                               max_factor=cfg_reg.nw_max_factor,
                               scale=cfg_reg.nw_scale)
    else:
        return None
