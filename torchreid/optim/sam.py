'''
Imported from: https://github.com/google-research/sam

Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 '''

import torch

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.rho = rho
        self.base_optimizer = base_optimizer
        defaults = dict(rho=rho, **self.base_optimizer.defaults)
        super().__init__(params, defaults)
        self.did_first_step = False

        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
<<<<<<< HEAD
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)
    
=======
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

>>>>>>> amp next
    @torch.no_grad()
    def second_step(self, grad_scale, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad: self.zero_grad()

<<<<<<< HEAD
    @torch.no_grad()
    def step(self):
        raise NotImplementedError("SAM doesn't work like the other optimizers,"
                                   " you should first call `first_step` and the `second_step`;")
=======
    def step(self, closure=None):
        raise NotImplementedError("SAM doesn't work like the other optimizers,"
                                   " you should first call `first_step` and the `second_step`;")

    # @torch.no_grad()
    # def first_step(self, zero_grad=False):
    #     grad_norm = self._grad_norm()
    #     if torch.isnan(grad_norm):
    #         grad_norm = torch.tensor(1., device=grad_norm.device)

    #     for group in self.param_groups:
    #         scale = self.rho / (grad_norm + 1e-12)

    #         for p in group["params"]:
    #             if not torch.sum(torch.isnan(p.grad)): continue
    #             e_w = p.grad * scale
    #             p.add_(e_w)  # climb to the local maximum "w + e(w)"
    #             self.state[p]["e_w"] = e_w

    #     if zero_grad: self.zero_grad()
    #     self.did_first_step = True

    # @torch.no_grad()
    # def second_step(self, grad_scale, zero_grad=False):
    #     for group in self.param_groups:
    #         for p in group["params"]:
    #             if not torch.sum(torch.isnan(p.grad)) or "e_w" not in self.state[p]: continue
    #             p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

    #     grad_scale.step(self.base_optimizer)
    #     if zero_grad: self.zero_grad()
    #     self.did_first_step = False

    # def step(self, zero_grad=False):
    #     if not self.did_first_step:
    #         raise NotImplementedError("SAM doesn't work like the other optimizers,"
    #                                " you should first call `first_step` and the `second_step`;")
    #     self.second_step(self, zero_grad=zero_grad)

    def _grad_norm(self):
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
>>>>>>> amp next
