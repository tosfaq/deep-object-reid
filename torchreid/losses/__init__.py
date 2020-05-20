from __future__ import division, print_function, absolute_import

from .am_softmax import AngleSimpleLinear, AMSoftmaxLoss
from .cross_entropy_loss import CrossEntropyLoss, PseudoCrossEntropyLoss
from .metric import MetricLosses, CenterLoss, MockTripletLoss, InvDistPushLoss
from .regularizers import ConvRegularizer, SVMORegularizer, NormRegularizer, get_regularizer
from .hard_mine_triplet_loss import TripletLoss
from .entropy import entropy, MaxEntropyLoss, MinEntropyLoss
from .kullback_leibler_div import kl_div, symmetric_kl_div, set_kl_div
from .tv_loss import TotalVarianceLoss


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
