from __future__ import division, print_function, absolute_import

from .am_softmax import AngleSimpleLinear, AMSoftmaxLoss
from .cross_entropy_loss import CrossEntropyLoss
from .metric import MetricLosses, CenterLoss, GlobalPushPlus
from .regularizers import ConvRegularizer, SVMORegularizer, get_regularizer
from .hard_mine_triplet_loss import TripletLoss
from .entropy import entropy, MaxEntropyLoss, MinEntropyLoss
from .kullback_leibler_div import kl_div, symmetric_kl_div, set_kl_div
from .metric_old import MetricLosses as MetricLossesOld


def DeepSupervision(criterion, xs, y, **kwargs):
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y, **kwargs)
    loss /= len(xs)

    return loss
