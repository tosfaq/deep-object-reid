from __future__ import print_function, absolute_import

import torch
from torch.optim.lr_scheduler import _LRScheduler
from bisect import bisect_right

AVAI_SCH = ['single_step', 'multi_step', 'cosine', 'multi_step_warmup']


def build_lr_scheduler(optimizer,
                       lr_scheduler='single_step',
                       stepsize=1,
                       gamma=0.1,
                       lr_scales=None,
                       max_epoch=1,
                       frozen=20,
                       warmup=10,
                       warmup_factor_base=0.1,
                       frozen_factor_base=0.1):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        lr_scheduler (str, optional): learning rate scheduler method. Default is single_step.
        stepsize (int or list, optional): step size to decay learning rate. When ``lr_scheduler``
            is "single_step", ``stepsize`` should be an integer. When ``lr_scheduler`` is
            "multi_step", ``stepsize`` is a list. Default is 1.
        gamma (float, optional): decay rate. Default is 0.1.
        max_epoch (int, optional): maximum epoch (for cosine annealing). Default is 1.

    Examples::
        >>> # Decay learning rate by every 20 epochs.
        >>> scheduler = torchreid.optim.build_lr_scheduler(
        >>>     optimizer, lr_scheduler='single_step', stepsize=20
        >>> )
        >>> # Decay learning rate at 30, 50 and 55 epochs.
        >>> scheduler = torchreid.optim.build_lr_scheduler(
        >>>     optimizer, lr_scheduler='multi_step', stepsize=[30, 50, 55]
        >>> )
    """
    if lr_scheduler not in AVAI_SCH:
        raise ValueError('Unsupported scheduler: {}. Must be one of {}'.format(lr_scheduler, AVAI_SCH))

    if lr_scheduler == 'single_step':
        if isinstance(stepsize, list):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                'For single_step lr_scheduler, stepsize must '
                'be an integer, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'multi_step':
        if not isinstance(stepsize, list):
            raise TypeError(
                'For multi_step lr_scheduler, stepsize must '
                'be a list, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch)
        )
    elif lr_scheduler == 'multi_step_warmup':
        if not isinstance(stepsize, list):
            raise TypeError(
                'For multi_step lr_scheduler, stepsize must '
                'be a list, but got {}'.format(type(stepsize))
            )

        scheduler = MultiStepLRWithWarmUp(
            optimizer, milestones=stepsize, frozen_iters=frozen, gamma=gamma, lr_scales=lr_scales,
            warmup_factor_base=warmup_factor_base, frozen_factor_base=frozen_factor_base, warmup_iters=warmup
        )
    else:
        raise ValueError('Unknown scheduler: {}'.format(lr_scheduler))

    return scheduler


class MultiStepLRWithWarmUp(_LRScheduler):
    def __init__(self,
                 optimizer,
                 milestones,
                 warmup_iters,
                 frozen_iters,
                 lr_scales=None,
                 warmup_method='linear',
                 warmup_factor_base=0.1,
                 frozen_factor_base=1.0,
                 gamma=0.1,
                 last_epoch=-1):
        if warmup_method not in {'constant', 'linear'}:
            raise KeyError('Unknown warm up method: {}'.format(warmup_method))

        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.lr_scales = lr_scales
        self.warmup_iters = warmup_iters
        self.frozen_iters = frozen_iters
        self.warmup_method = warmup_method
        self.warmup_factor_base = warmup_factor_base
        self.frozen_factor_base = frozen_factor_base

        self.uses_lr_scales = self.lr_scales is not None and len(self.lr_scales) > 0
        if self.uses_lr_scales:
            assert len(self.lr_scales) == len(self.milestones) + 1

        # Base class calls method `step` which increases `last_epoch` by 1 and then calls
        # method `get_lr` with this value. If `last_epoch` is not equal to -1, we drop
        # the first step, so to avoid this dropping do small fix by subtracting 1
        if last_epoch > -1:
            last_epoch = last_epoch - 1
        elif last_epoch < -1:
            raise ValueError('Learning rate scheduler got incorrect parameter last_epoch = {}'.format(last_epoch))

        super(MultiStepLRWithWarmUp, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # During warm up change learning rate on every step according to warmup_factor
        if self.last_epoch < self.frozen_iters:
            return [self.frozen_factor_base * base_lr for base_lr in self.base_lrs]
        if self.last_epoch < self.frozen_iters + self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor_base
            elif self.warmup_method == 'linear':
                alpha = (self.last_epoch - self.frozen_iters) / self.warmup_iters
                warmup_factor = self.warmup_factor_base * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        # On the last step of warm up set learning rate equal to base LR
        elif self.last_epoch == self.frozen_iters + self.warmup_iters:
            return [base_lr for base_lr in self.base_lrs]
        # After warm up increase LR according to defined in `milestones` values of steps
        else:
            if self.uses_lr_scales:
                lr_scale = self.lr_scales[bisect_right(self.milestones, self.last_epoch)]
            else:
                lr_scale = self.gamma ** bisect_right(self.milestones, self.last_epoch)

            return [base_lr * lr_scale for base_lr in self.base_lrs]

    def __repr__(self):
        format_string = self.__class__.__name__ + \
                        '[warmup_method = {}, warmup_factor_base = {}, warmup_iters = {},' \
                        ' milestones = {}, gamma = {}]'.format(self.warmup_method, self.warmup_factor_base,
                                                               self.warmup_iters, str(list(self.milestones)),
                                                               self.gamma)
        return format_string
