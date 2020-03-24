from __future__ import division, absolute_import

__all__ = ['AverageMeter']


class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, enable_zeros=False):
        self.enable_zeros = enable_zeros

        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        if n > 0:
            if self.enable_zeros:
                self._update(val, n)
            elif val > 0.0:
                self._update(val, n)

    def _update(self, val, n):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
