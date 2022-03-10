# Copyright 2020 Google Research
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

'''
Imported from: https://github.com/google-research/sam
'''

import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=True, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.base_optimizer = base_optimizer
        defaults = dict(rho=rho, adaptive=adaptive, **self.base_optimizer.defaults)

        super().__init__(params, defaults)
        self.rho = rho
        self.adaptive = adaptive
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        if self._has_overflow(self.param_groups):
            if zero_grad: self.zero_grad()
            return True

        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()
        return False

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        if self._has_overflow(self.param_groups):
            if zero_grad: self.zero_grad()
            return

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self):
        raise NotImplementedError("SAM doesn't work like the other optimizers,"
                                   " you should first call `first_step` and the `second_step`;")

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    @staticmethod
    def _has_overflow(params):
        ''' Check whether the gradient overflow occurred in model parameters '''
        def _has_inf_or_nan(x):
            try:
                # if x is half, the .float() incurs an additional deep copy, but it's necessary if
                # Pytorch's .sum() creates a one-element tensor of the same type as x
                # (which is true for some recent version of pytorch).
                cpu_sum = float(x.float().sum())
                # More efficient version that can be used if .sum() returns a Python scalar
                # cpu_sum = float(x.sum())
            except RuntimeError as instance:
                # We want to check if inst is actually an overflow exception.
                # RuntimeError could come from a different error.
                # If so, we still want the exception to propagate.
                if "value cannot be converted" not in instance.args[0]:
                    raise
                return True
            else:
                if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                    return True
                return False

        for group in params:
            for p in group["params"]:
                if p.grad is not None and _has_inf_or_nan(p.grad.data):
                    return True

        return False
