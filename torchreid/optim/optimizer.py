from __future__ import absolute_import, print_function
import warnings

import torch
import torch.nn as nn

from .radam import RAdam
from .sam import SAM

AVAI_OPTIMS = {'adam', 'amsgrad', 'sgd', 'rmsprop', 'radam', 'sam'}


class OtimizerBuilder:
    def __init__(self,
                model,
                optim='adam',
                base_optim = 'sgd',
                lr=0.0003,
                weight_decay=5e-04,
                momentum=0.9,
                sgd_dampening=0,
                sgd_nesterov=False,
                rmsprop_alpha=0.99,
                adam_beta1=0.9,
                adam_beta2=0.99,
                staged_lr=False,
                new_layers='',
                base_lr_mult=0.1,
                sam_rho = 0.05):

        """A class for building an optimizer.

        Args:
            model (nn.Module): model.
            optim (str, optional): optimizer. Default is "adam".
            lr (float, optional): learning rate. Default is 0.0003.
            weight_decay (float, optional): weight decay (L2 penalty). Default is 5e-04.
            momentum (float, optional): momentum factor in sgd. Default is 0.9,
            sgd_dampening (float, optional): dampening for momentum. Default is 0.
            sgd_nesterov (bool, optional): enables Nesterov momentum. Default is False.
            rmsprop_alpha (float, optional): smoothing constant for rmsprop. Default is 0.99.
            adam_beta1 (float, optional): beta-1 value in adam. Default is 0.9.
            adam_beta2 (float, optional): beta-2 value in adam. Default is 0.99,
            staged_lr (bool, optional): uses different learning rates for base and new layers. Base
                layers are pretrained layers while new layers are randomly initialized, e.g. the
                identity classification layer. Enabling ``staged_lr`` can allow the base layers to
                be trained with a smaller learning rate determined by ``base_lr_mult``, while the new
                layers will take the ``lr``. Default is False.
            new_layers (str or list): attribute names in ``model``. Default is empty.
            base_lr_mult (float, optional): learning rate multiplier for base layers. Default is 0.1.

        Examples::
            >>> # A normal optimizer can be built by
            >>> optimizer = torchreid.optim.build_optimizer(model, optim='sgd', lr=0.01)
            >>> # If you want to use a smaller learning rate for pretrained layers
            >>> # and the attribute name for the randomly initialized layer is 'classifier',
            >>> # you can do
            >>> optimizer = torchreid.optim.build_optimizer(
            >>>     model, optim='sgd', lr=0.01, staged_lr=True,
            >>>     new_layers='classifier', base_lr_mult=0.1
            >>> )
            >>> # Now the `classifier` has learning rate 0.01 but the base layers
            >>> # have learning rate 0.01 * 0.1.
            >>> # new_layers can also take multiple attribute names. Say the new layers
            >>> # are 'fc' and 'classifier', you can do
            >>> optimizer = torchreid.optim.build_optimizer(
            >>>     model, optim='sgd', lr=0.01, staged_lr=True,
            >>>     new_layers=['fc', 'classifier'], base_lr_mult=0.1
            >>> )
        """
        self.model = model
        self.optim = optim
        self.base_optim = base_optim
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.sgd_dampening = sgd_dampening
        self.sgd_nesterov = sgd_nesterov
        self.rmsprop_alpha = rmsprop_alpha
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.staged_lr = staged_lr
        self.new_layers = new_layers
        self.base_lr_mult = base_lr_mult
        self.sam_rho = sam_rho

    def build_optimizer(self):
        if self.optim == 'sam':
            base_optim = self._build_optim(optim=self.base_optim)
            optimizer = self._build_optim(self.optim, base_optim=base_optim)
        else:
            optimizer = self._build_optim(optim=self.optim)

        return optimizer

    def _build_optim(self, optim, base_optim=None):

        if optim not in AVAI_OPTIMS:
            raise ValueError(
                'Unsupported optim: {}. Must be one of {}'.format(
                    optim, AVAI_OPTIMS
                )
            )

        if not isinstance(self.model, nn.Module):
            raise TypeError(
                'model given to build_optimizer must be an instance of nn.Module'
            )

        if self.staged_lr:
            if isinstance(self.new_layers, str):
                if self.new_layers is None:
                    warnings.warn(
                        'new_layers is empty, therefore, staged_lr is useless'
                    )
                self.new_layers = [self.new_layers]

            if isinstance(self.model, nn.DataParallel):
                self.model = self.model.module

            base_params = []
            base_layers = []
            new_params = []

            for name, module in self.model.named_children():
                if name in self.new_layers:
                    new_params += [p for p in module.parameters()]
                else:
                    base_params += [p for p in module.parameters()]
                    base_layers.append(name)

            param_groups = [
                {
                    'params': base_params,
                    'lr': self.lr * self.base_lr_mult
                },
                {
                    'params': new_params
                },
            ]

        else:
            param_groups = self.model.parameters()
        if optim == 'adam':
            optimizer = torch.optim.Adam(
                param_groups,
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(self.adam_beta1, self.adam_beta2),
            )

        elif optim == 'amsgrad':
            optimizer = torch.optim.Adam(
                param_groups,
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(self.adam_beta1, self.adam_beta2),
                amsgrad=True,
            )

        elif optim == 'sgd':
            optimizer = torch.optim.SGD(
                param_groups,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                dampening=self.sgd_dampening,
                nesterov=self.sgd_nesterov,
            )

        elif optim == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                param_groups,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                alpha=self.rmsprop_alpha,
            )

        elif optim == 'radam':
            optimizer = RAdam(
                param_groups,
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(self.adam_beta1, self.adam_beta2)
            )

        if optim == 'sam':
            if not base_optim:
                raise ValueError("SAM cannot operate without base optimizer. "
                                 "Please add it to configuration file")
            optimizer = SAM(
                params=param_groups,
                base_optimizer=base_optim,
                rho=self.sam_rho
            )

        return optimizer
