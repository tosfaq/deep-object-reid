# Copyright (c) 2018-2021 Kaiyang Zhou
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import absolute_import, division, print_function
import copy
from importlib.metadata import requires

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast

from torchreid import metrics
from torchreid.engine import Engine
from torchreid.utils import get_model_attr, sample_mask
from torchreid.losses import (AMSoftmaxLoss, CrossEntropyLoss)
from torchreid.optim import SAM


class ImageAMSoftmaxEngine(Engine):
    r"""AM-Softmax-loss engine for image-reid.
    """
    def __init__(self, datamanager, models, optimizers, schedulers, use_gpu, save_all_chkpts,
                 train_patience, early_stopping, lr_decay_factor, loss_name, label_smooth,
                 margin_type, aug_type, decay_power, alpha, size, lr_finder, aug_prob,
                 conf_penalty, pr_product, m, clip_grad, symmetric_ce, enable_rsc,
                 should_freeze_aux_models, nncf_metainfo, compression_ctrl, initial_lr,
                 target_metric, use_ema_decay, ema_decay, mix_precision, **kwargs):
        super(ImageAMSoftmaxEngine, self).__init__(datamanager,
                                                   models=models,
                                                   optimizers=optimizers,
                                                   schedulers=schedulers,
                                                   use_gpu=use_gpu,
                                                   save_all_chkpts=save_all_chkpts,
                                                   train_patience=train_patience,
                                                   lr_decay_factor=lr_decay_factor,
                                                   early_stopping=early_stopping,
                                                   should_freeze_aux_models=should_freeze_aux_models,
                                                   nncf_metainfo=nncf_metainfo,
                                                   compression_ctrl=compression_ctrl,
                                                   initial_lr=initial_lr,
                                                   target_metric=target_metric,
                                                   lr_finder=lr_finder,
                                                   use_ema_decay=use_ema_decay,
                                                   ema_decay=ema_decay)

        assert loss_name in ['softmax', 'am_softmax']
        self.loss_name = loss_name
        if loss_name == 'am_softmax':
            assert m >= 0.0

        self.clip_grad = clip_grad
        self.enable_rsc = enable_rsc
        self.enable_sam = isinstance(self.optims[self.main_model_name], SAM)
        for model_name in self.get_model_names():
            assert isinstance(self.optims[model_name], SAM) == self.enable_sam, "SAM must be enabled \
                                                                                 for all models or none of them"
        self.aug_type = aug_type
        self.aug_prob = aug_prob
        self.aug_index = None
        self.lam = None
        self.alpha = alpha
        self.decay_power = decay_power
        self.size =  size
        self.prev_smooth_metric = 0.
        self.mix_precision = mix_precision
        self.scaler = GradScaler(enabled=mix_precision)

        self.num_classes = self.datamanager.num_train_ids
        self.ml_losses = list()
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        if loss_name == 'softmax':
            self.main_loss = CrossEntropyLoss(
                use_gpu=self.use_gpu,
                label_smooth=label_smooth,
                augmentations=self.aug_type,
                conf_penalty=conf_penalty,
                scale=self.am_scale
            )
        elif loss_name == 'am_softmax':
            self.main_loss = AMSoftmaxLoss(
                use_gpu=self.use_gpu,
                label_smooth=label_smooth,
                margin_type=margin_type,
                aug_type=aug_type,
                conf_penalty=conf_penalty,
                m=m,
                s=self.am_scale,
                pr_product=pr_product,
                symmetric_ce=symmetric_ce,
            )


    @staticmethod
    def _valid(value):
        return value is not None and value > 0

    def forward_backward(self, data):
        n_iter = self.epoch * self.num_batches + self.batch_idx
        imgs, targets = self.parse_data_for_train(data, self.use_gpu)

        imgs = self._apply_batch_augmentation(imgs)
        model_names = self.get_model_names()
        num_models = len(model_names)

        steps = [1, 2] if self.enable_sam and not self.lr_finder else [1]
        for step in steps:
            # if sam is enabled then statistics will be written each step, but will be saved only the second time
            # this is made just for convenience
            loss_summary = dict()
            models_logits = [[] for i in range(num_models)]

            for i, model_name in enumerate(model_names):
                loss, model_loss_summary, acc, scaled_logits = self._single_model_losses(
                    self.models[model_name], imgs, targets, n_iter, model_name
                    )
                models_logits[i] = scaled_logits
                loss_summary.update(model_loss_summary)
                if i == 0: # main model
                    main_acc = acc

            for i, model_name in enumerate(model_names):
                self.optims[model_name].zero_grad()
                if num_models > 1: # mutual learning
                    mutual_loss = 0
                    for j in range(num_models):
                        if i != j:
                            with torch.no_grad():
                                trg_probs = F.softmax(models_logits[j], dim=1)
                            mutual_loss += self.loss_kl(F.log_softmax(models_logits[i], dim = 1),
                                                    trg_probs)
                    loss_summary[f'mutual_{model_names[i]}'] = mutual_loss.item()

                    should_turn_off_mutual_learning = self._should_turn_off_mutual_learning(self.epoch)
                    coeff_mutual_learning = int(not should_turn_off_mutual_learning)

                    loss += coeff_mutual_learning * mutual_loss / (num_models - 1)

                if self.compression_ctrl:
                    compression_loss = self.compression_ctrl.loss()
                    loss_summary['compression_loss'] = compression_loss
                    loss += compression_loss

                # backward pass
                self.scaler.scale(loss).backward()
                if not self.models[model_name].training:
                    continue

                if self.clip_grad != 0 and step == 1:
                    self.scaler.unscale_(self.optims[model_name])
                    torch.nn.utils.clip_grad_norm_(self.models[model_name].parameters(), self.clip_grad)
                if not self.enable_sam and step == 1:
                    self.scaler.step(self.optims[model_name])
                    self.scaler.update()
                elif step == 1:
                    assert self.enable_sam
                    if self.clip_grad == 0:
                        # if self.clip_grad == 0  this means that unscale_ wasn't applied,
                        # so we manually unscale the parameters to perform SAM manipulations
                        self.scaler.unscale_(self.optims[model_name])
                    overflow = self.optims[model_name].first_step()
                    self.scaler.update() # update scaler after first step
                    if overflow:
                        print("Overflow occurred. Skipping step ...")
                        loss_summary['loss'] = loss.item()
                        # skip second step  if overflow occurred
                        return loss_summary, main_acc
                else:
                    assert self.enable_sam and step==2
                    # unscale the parameters to perform SAM manipulations
                    self.scaler.unscale_(self.optims[model_name])
                    self.optims[model_name].second_step()
                    self.scaler.update()

            loss_summary['loss'] = loss.item()

        return loss_summary, main_acc

    def _single_model_losses(self, model, imgs, targets, n_iter, model_name):
        run_kwargs = dict()
        if self.enable_rsc:
            run_kwargs['gt_labels'] = targets

        with autocast(enabled=self.mix_precision):
            model_output = model(imgs, **run_kwargs)
            all_logits = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
            loss_summary = dict()
            acc = 0
            trg_num_samples = targets.numel()
            if trg_num_samples == 0:
                raise RuntimeError("There is no samples in a batch!")

            loss = self.main_loss(all_logits, targets, aug_index=self.aug_index,
                                                lam=self.lam, scale=self.scales[model_name])
            acc += metrics.accuracy(all_logits, targets)[0].item()
            loss_summary[f'main_{model_name}'] = loss.item()

            scaled_logits = self.main_loss.get_scale() * all_logits

            return loss, loss_summary, acc, scaled_logits


    def _apply_batch_augmentation(self, imgs):
        def rand_bbox(size, lam):
            W = size[2]
            H = size[3]
            cut_rat = np.sqrt(1. - lam)
            cut_w = np.int(W * cut_rat)
            cut_h = np.int(H * cut_rat)

            # uniform
            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

            return bbx1, bby1, bbx2, bby2

        if self.aug_type == 'fmix':
            r = np.random.rand(1)
            if self.alpha > 0 and r[0] <= self.aug_prob:
                lam, fmask = sample_mask(self.alpha, self.decay_power, self.size)
                index = torch.randperm(imgs.size(0), device=imgs.device)
                fmask = torch.from_numpy(fmask).float().to(imgs.device)
                # Mix the images
                x1 = fmask * imgs
                x2 = (1 - fmask) * imgs[index]
                self.aug_index = index
                self.lam = lam
                imgs = x1 + x2
            else:
                self.aug_index = None
                self.lam = None

        elif self.aug_type == 'mixup':
            r = np.random.rand(1)
            if self.alpha > 0 and r <= self.aug_prob:
                lam = np.random.beta(self.alpha, self.alpha)
                index = torch.randperm(imgs.size(0), device=imgs.device)

                imgs = lam * imgs + (1 - lam) * imgs[index, :]
                self.lam = lam
                self.aug_index = index
            else:
                self.aug_index = None
                self.lam = None

        elif self.aug_type == 'cutmix':
            r = np.random.rand(1)
            if self.alpha > 0 and r <= self.aug_prob:
                # generate mixed sample
                lam = np.random.beta(self.alpha, self.alpha)
                rand_index = torch.randperm(imgs.size(0), device=imgs.device)

                bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
                imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
                self.lam = lam
                self.aug_index = rand_index
            else:
                self.aug_index = None
                self.lam = None

        return imgs


    def exit_on_plateau_and_choose_best(self, accuracy):
        '''
        The function returns a pair (should_exit, is_candidate_for_best).

        The function sets this checkpoint as a candidate for best if either it is the first checkpoint
        for this LR or this checkpoint is better then the previous best.

        The function sets should_exit = True if the LR is the minimal allowed
        LR (i.e. self.lb_lr) and the best checkpoint is not changed for self.train_patience
        epochs.
        '''

        # Note that we take LR of the previous iter, not self.get_current_lr(),
        # since typically the method exit_on_plateau_and_choose_best is called after
        # the method update_lr, so LR drop happens before.
        # If we had used the method self.get_current_lr(), the last epoch
        # before LR drop would be used as the first epoch with the new LR.
        should_exit = False
        is_candidate_for_best = False
        current_metric = np.round(accuracy, 4)
        if self.best_metric >= current_metric:
            # one drop has been done -> start early stopping
            if round(self.current_lr, 8) < round(self.initial_lr, 8) and self.warmup_finished:
                self.iter_to_wait += 1
                if self.iter_to_wait >= self.train_patience:
                    print("LOG:: The training should be stopped due to no improvements for {} epochs".format(self.train_patience))
                    should_exit = True
        else:
            self.best_metric = current_metric
            self.iter_to_wait = 0
            is_candidate_for_best = True

        return should_exit, is_candidate_for_best
