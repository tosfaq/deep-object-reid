# Copyright (c) 2018-2021 Kaiyang Zhou
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import absolute_import, division, print_function

from .am_softmax import AMSoftmaxLoss, AngleSimpleLinear
from .asl import AsymmetricLoss, AMBinaryLoss
from .cross_entropy_loss import CrossEntropyLoss, PseudoCrossEntropyLoss
from .metric_losses import CenterLoss, MetricLosses


def DeepSupervision(criterion, xs, y, **kwargs):
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    total_loss = 0.0
    num_losses = 0
    for x in xs:
        loss_val = criterion(x, y, **kwargs)
        if loss_val > 0.0:
            total_loss += loss_val
            num_losses += 1
    total_loss /= float(num_losses if num_losses > 0 else 1)

    return total_loss
