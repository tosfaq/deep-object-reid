import torch
from torch import nn
import torch.nn.functional as F

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AMBinaryLoss(nn.Module):
    def __init__(self, m=0.35, k=0.8, t=1, s=30,
                eps=1e-8, label_smooth=0., gamma_neg=0,
                gamma_pos=0, probability_margin=0.05):

        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.asymmetric_focus = gamma_neg > 0 or gamma_pos > 0
        self.eps = eps
        self.label_smooth = label_smooth
        self.m = m
        self.k = k
        self.s = s
        self.clip = probability_margin

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def get_last_scale(self):
        return self.s

    def forward(self, cos_theta, targets, scale=None):
        """"
        Parameters
        ----------
        cos_theta: dot product between normalized features and proxies
        targets: targets (multi-label binarized vector)
        """
        self.s = scale if scale else self.s
        if self.label_smooth > 0:
            targets = targets * (1 - self.label_smooth)
            targets[targets == 0] = self.label_smooth
        self.anti_targets = 1 - targets

        if self.asymmetric_focus:
            # Calculating Probabilities
            xs_pos = torch.sigmoid(self.s * (cos_theta - self.m))
            xs_neg = torch.sigmoid(- self.s * (cos_theta + self.m))
            # Asymmetric Probability Shifting
            if self.clip is not None and self.clip > 0:
                xs_neg = (xs_neg + self.clip).clamp(max=1)
            pt0 = xs_neg * targets
            pt1 = xs_pos * (1 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            # P_pos ** gamm_neg * (...) + P_neg ** gamma_pos * (...)
            one_sided_w = torch.pow(pt, one_sided_gamma)
            balance_koeff_pos = self.k / self.s
            balance_koeff_neg = (1 - self.k) / self.s
        else:
            assert not self.asymmetric_focus
            # SphereFace2 balancing coefficients
            balance_koeff_pos = self.k / self.s
            balance_koeff_neg = (1 - self.k) / self.s

        self.loss = balance_koeff_pos * targets * F.logsigmoid(self.s * (cos_theta - self.m))
        self.loss.add_(balance_koeff_neg * self.anti_targets * F.logsigmoid(- self.s * (cos_theta + self.m)))

        if self.asymmetric_focus:
            self.loss *= one_sided_w

        return - self.loss.sum()