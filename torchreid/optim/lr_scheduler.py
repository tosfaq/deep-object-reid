from __future__ import print_function, absolute_import

import torch
import math
from torch.optim.lr_scheduler import _LRScheduler
from torch import optim
from bisect import bisect_right

AVAI_SCH = ['single_step', 'multi_step', 'cosine', 'multi_step_warmup', 'cosine_warmup', 'cosine_cycle', 'reduce_on_plateau']


def build_lr_scheduler(optimizer,
                       lr_scheduler='single_step',
                       stepsize=1,
                       gamma=0.1,
                       lr_scales=None,
                       max_epoch=1,
                       frozen=20,
                       warmup=10,
                       multiplier = 10,
                       first_cycle_steps=10,
                       cycle_mult = 1.,
                       min_lr = 1e-4,
                       max_lr = 0.1,
                       patience = 5,
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

        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'multi_step':
        if not isinstance(stepsize, list):
            raise TypeError(
                'For multi_step lr_scheduler, stepsize must '
                'be a list, but got {}'.format(type(stepsize))
            )

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'cosine':
        print("NOT HERE")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch)
        )
    elif lr_scheduler == 'multi_step_warmup':
        print("HERE 2")
        if not isinstance(stepsize, list):
            raise TypeError(
                'For multi_step lr_scheduler, stepsize must '
                'be a list, but got {}'.format(type(stepsize))
            )

        scheduler = MultiStepLRWithWarmUp(
            optimizer, milestones=stepsize, frozen_iters=frozen, gamma=gamma, lr_scales=lr_scales,
            warmup_factor_base=warmup_factor_base, frozen_factor_base=frozen_factor_base, warmup_iters=warmup
        )
    elif lr_scheduler == 'cosine_warmup':
        scheduler_after = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch)
        )
        scheduler = WarmupScheduler(optimizer, multiplier=multiplier, total_epoch=warmup, after_scheduler=scheduler_after)
    elif lr_scheduler == 'cosine_cycle':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=first_cycle_steps, mode='triangular2')
        # scheduler = CosineAnnealingCycleRestart(optimizer, first_cycle_steps=first_cycle_steps, cycle_mult=cycle_mult,
        # max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup, gamma=gamma)
    elif lr_scheduler == 'reduce_on_plateau':
        scheduler = scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, patience=patience, verbose=True, min_lr=min_lr)
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

class CosineAnnealingCycleRestart(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer : optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle

        super().__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)
        # scale down initial learning rate
        self.init_lr()

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= self.multiplier
            self.base_lrs.append(param_group['lr'])

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super().step(epoch)

class ReduceOnPlateau(optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)

    def get_lr_lower_bound(self):
        return self.min_lrs
