from __future__ import division, absolute_import
import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    r"""Cross entropy loss with label smoothing regularizer.
    """

    def __init__(self, epsilon=0.1, use_gpu=True, label_smooth=True, conf_penalty=None):
        super(CrossEntropyLoss, self).__init__()
        self.epsilon = epsilon if label_smooth else 0
        self.use_gpu = use_gpu
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.conf_penalty = conf_penalty

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """

        log_probs = self.log_softmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()

        num_classes = inputs.size(1)
        targets = (1.0 - self.epsilon) * targets + self.epsilon / float(num_classes)
        sm_loss = (- targets * log_probs).sum(dim=1)

        if self.conf_penalty is not None and self.conf_penalty > 0.0:
            probs = self.softmax(inputs)
            entropy = (-probs * torch.log(probs.clamp(min=1e-12))).sum(dim=1)

            losses = sm_loss - self.conf_penalty * entropy
            losses = losses[losses > 0.0]

            return losses.mean() if losses.numel() > 0 else losses.sum()

        return sm_loss.mean()
