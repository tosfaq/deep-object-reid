from __future__ import division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    r"""Cross entropy loss with label smoothing regularizer.

    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

    With label smoothing, the label :math:`y` for a class is computed by

    .. math::
        \begin{equation}
        (1 - \eps) \times y + \frac{\eps}{K},
        \end{equation}

    where :math:`K` denotes the number of classes and :math:`\eps` is a weight. When
    :math:`\eps = 0`, the loss function reduces to the normal cross entropy.

    Args:
        num_classes (int): number of classes.
        eps (float, optional): weight. Default is 0.1.
        use_gpu (bool, optional): whether to use gpu devices. Default is True.
        label_smooth (bool, optional): whether to apply label smoothing. Default is True.
    """

    def __init__(self, scale=1.0, epsilon=0.1, use_gpu=True, label_smooth=True,
                 conf_penalty=None, penalty_scale=5.0):
        super(CrossEntropyLoss, self).__init__()

        self.scale = scale
        self.epsilon = epsilon if label_smooth else 0
        self.use_gpu = use_gpu
        self.conf_penalty = conf_penalty
        self.penalty_scale = penalty_scale

    def forward(self, inputs, targets, scale=None, iteration=None):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """

        if inputs.size(0) == 0:
            return torch.zeros([], dtype=inputs.dtype, device=inputs.device)

        scale = scale if scale is not None and scale > 0.0 else self.scale

        scaled_inputs = scale * inputs
        log_probs = F.log_softmax(scaled_inputs, dim=1)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()

        num_classes = inputs.size(1)
        targets = (1.0 - self.epsilon) * targets + self.epsilon / float(num_classes)
        sm_loss = (- targets * log_probs).sum(dim=1)

        if self.conf_penalty is not None and self.conf_penalty > 0.0\
           and self.penalty_scale is not None and self.penalty_scale > 0.0:
            probs = F.softmax(scaled_inputs, dim=1)
            entropy = (-probs * torch.log(probs.clamp(min=1e-12))).sum(dim=1)

            losses = self.penalty_scale * sm_loss - self.conf_penalty * entropy
            losses = losses[losses > 0.0]

            return losses.mean() if losses.numel() > 0 else losses.sum()

        return sm_loss.mean()


class PseudoCrossEntropyLoss(nn.Module):
    def __init__(self, scale=1.0, epsilon=0.1, use_gpu=True, label_smooth=True, conf_penalty=None):
        super(PseudoCrossEntropyLoss, self).__init__()

        self.scale = scale
        self.epsilon = epsilon if label_smooth else 0
        self.use_gpu = use_gpu
        self.conf_penalty = conf_penalty

    def forward(self, inputs, scale=None):
        if inputs.size(0) == 0:
            return torch.zeros([], dtype=inputs.dtype, device=inputs.device)

        scale = scale if scale is not None and scale > 0.0 else self.scale
        targets = torch.argmax(inputs, dim=-1)

        log_probs = F.log_softmax(scale * inputs, dim=1)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()

        num_classes = inputs.size(1)
        targets = (1.0 - self.epsilon) * targets + self.epsilon / float(num_classes)
        sm_loss = (- targets * log_probs).sum(dim=1)

        if self.conf_penalty is not None and self.conf_penalty > 0.0:
            probs = F.softmax(scale * inputs, dim=1)
            entropy = (-probs * torch.log(probs.clamp(min=1e-12))).sum(dim=1)

            losses = sm_loss - self.conf_penalty * entropy
            losses = losses[losses > 0.0]

            return losses.mean() if losses.numel() > 0 else losses.sum()

        return sm_loss.mean()
