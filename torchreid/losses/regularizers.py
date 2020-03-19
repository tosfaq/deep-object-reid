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
    def __init__(self, reg_class, controller):
        super().__init__()
        self.reg_instance = reg_class(controller)

    def forward(self, net):

        accumulator = torch.tensor(0.0).cuda()
        for module in net.module.modules():
            if not isinstance(module, nn.Conv2d):
                continue

            accumulator += self.reg_instance(module.weight)

        return accumulator


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


def get_regularizer(cfg_reg):
    if cfg_reg.ow:
        return ConvRegularizer(SVMORegularizer, cfg_reg.ow_beta)
    else:
        return None
