from __future__ import absolute_import, division, print_function
from enum import auto

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from torchreid import metrics
from torchreid.losses import AsymmetricLoss, AMBinaryLoss
from torchreid.metrics.accuracy import accuracy
from torchreid.optim import SAM
from torchreid.utils import get_model_attr
from torchreid.engine import Engine

class MultilabelEngine(Engine):
    r"""Multilabel classification engine. It supports ASL, BCE and Angular margin loss with binary classification."""
    def __init__(self, datamanager, models, optimizers, schedulers, use_gpu, save_all_chkpts,
                 train_patience, early_stopping, lr_decay_factor, loss_name, label_smooth,
                 lr_finder, m, amb_k, amb_t, clip_grad,
                 should_freeze_aux_models, nncf_metainfo, compression_ctrl, initial_lr,
                 target_metric, use_ema_decay, ema_decay, asl_gamma_pos, asl_gamma_neg, asl_p_m,
                 mix_precision, **kwargs):

        super().__init__(datamanager,
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
                        lr_finder=lr_finder,
                        target_metric=target_metric,
                        use_ema_decay=use_ema_decay,
                        ema_decay=ema_decay)

        self.main_losses = nn.ModuleList()
        self.clip_grad = clip_grad
        self.num_classes = self.datamanager.num_train_ids
        self.aug_index = None
        self.lam = None

        if loss_name == 'asl':
            self.main_losses = AsymmetricLoss(
                gamma_neg=asl_gamma_neg,
                gamma_pos=asl_gamma_pos,
                probability_margin=asl_p_m,
                label_smooth=label_smooth,
            )
        elif loss_name == 'bce':
            self.main_losses = AsymmetricLoss(
                gamma_neg=0,
                gamma_pos=0,
                probability_margin=0,
                label_smooth=label_smooth,
            )
        elif loss_name == 'am_binary':
            self.main_losses = AMBinaryLoss(
                m=m,
                k=amb_k,
                t=amb_t,
                s=self.am_scale,
                gamma_neg=asl_gamma_neg,
                gamma_pos=asl_gamma_pos,
                label_smooth=label_smooth,
            )


        self.enable_sam = isinstance(self.optims[self.main_model_name], SAM)

        for model_name in self.get_model_names():
            assert isinstance(self.optims[model_name], SAM) == self.enable_sam, "SAM must be enabled \
                                                                                 for all models or none of them"
        self.scaler = GradScaler(enabled=mix_precision)
        self.mix_precision = mix_precision
        self.prev_smooth_accuracy = 0.

    def forward_backward(self, data):
        imgs, targets = self.parse_data_for_train(data, use_gpu=self.use_gpu)

        model_names = self.get_model_names()
        num_models = len(model_names)
        steps = [1,2] if self.enable_sam and not self.lr_finder else [1]
        # forward pass
        for step in steps:
            # if sam is enabled then statistics will be written each step, but will be saved only the second time
            # this is made just for convenience
            loss_summary = dict()
            models_logits = [[] for i in range(num_models)]
            num_models = len(self.models)

            for i, model_name in enumerate(model_names):
                loss, model_loss_summary, acc, scaled_logits = self._single_model_losses(
                    self.models[model_name], imgs, targets, model_name
                    )
                models_logits[i] = scaled_logits
                loss_summary.update(model_loss_summary)
                if i == 0: # main model
                    main_acc = acc

            for i, model_name in enumerate(model_names):
                self.optims[model_name].zero_grad()
                # compute mutual loss
                if len(model_names) > 1:
                    mutual_loss = 0
                    probabilities_i = torch.sigmoid(models_logits[i])
                    for j, model_name in enumerate(model_names):
                        if i != j:
                            with torch.no_grad():
                                probabilities_j = torch.sigmoid(models_logits[j])
                            mutual_loss += self.kl_div_binary(probabilities_i, probabilities_j)
                    loss_summary[f'mutual_{model_names[i]}'] = mutual_loss.item()

                    should_turn_off_mutual_learning = self._should_turn_off_mutual_learning(self.epoch)
                    coeff_mutual_learning = int(not should_turn_off_mutual_learning)

                    loss += coeff_mutual_learning * mutual_loss / (num_models - 1)

                    if self.compression_ctrl:
                        compression_loss = self.compression_ctrl.loss()
                        loss_summary['compression_loss'] = compression_loss
                        loss += compression_loss

                # backward pass
                self.scaler.scale(loss).backward(retain_graph=False)
                if self.clip_grad != 0 and step == 1:
                    self.scaler.unscale_(self.optims[model_name])
                    torch.nn.utils.clip_grad_norm_(self.models[model_name].parameters(), self.clip_grad)
                if not self.enable_sam and step == 1:
                    self.scaler.step(self.optims[model_name])
                    self.scaler.update()
                elif step == 1:
                    assert self.enable_sam
                    if self.clip_grad == 0:
                        # if self.clip_grad == 0  this means that unscale_ wasn't applied
                        # unscale parameters to perform SAM manipulations with parameters
                        self.scaler.unscale_(self.optims[model_name])
                    overflow = self.optims[model_name].first_step(self.scaler)
                    self.scaler.update() # update scaler after first step
                    if overflow:
                        print("Overflow occurred. Skipping step ...")
                        loss_summary['loss'] = loss.item()
                        # skip second step  if overflow occurred
                        return loss_summary, main_acc
                else:
                    assert self.enable_sam and step==2
                    if self.clip_grad == 0:
                        self.scaler.unscale_(self.optims[model_name])
                    self.optims[model_name].second_step()
                    self.scaler.update()

            loss_summary['loss'] = loss.item()

        return loss_summary, main_acc

    def _single_model_losses(self, model, imgs, targets, model_name):
        with autocast(enabled=self.mix_precision):
            model_output = model(imgs)
            all_logits = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
            loss_summary = dict()
            acc = 0
            trg_num_samples = targets.numel()
            if trg_num_samples == 0:
                raise RuntimeError("There is no samples in a batch!")

            loss = self.main_losses(all_logits, targets, aug_index=self.aug_index,
                                    lam=self.lam, scale=self.scales[model_name])
            acc += metrics.accuracy_multilabel(all_logits, targets).item()
            loss_summary[f'main_{model_name}'] = loss.item()

            scaled_logits = self.scales[model_name] * all_logits

        return loss, loss_summary, acc, scaled_logits

    def kl_div_binary(self, x, y):
        ''' compute KL divergence between two tensors represented
        independent binary distributions'''
        # get binary distributions for two models with shape = (BxCx2)
        p = torch.stack((x, (1-x))).permute(1,2,0)
        q = torch.stack((y, (1-y))).permute(1,2,0)
        # log probabilities
        p_log = torch.log(p.add_(1e-8))
        # compute true KLDiv for each sample, than do the batchmean reduction
        return F.kl_div(p_log, q, reduction='none').sum(2).div_(x.size(1)).sum().div_(x.size(0))

    def exit_on_plateau_and_choose_best(self, accuracy):
        '''
        The function returns a pair (should_exit, is_candidate_for_best).

        The function sets this checkpoint as a candidate for best if either it is the first checkpoint
        for this LR or this checkpoint is better then the previous best.

        The function sets should_exit = True if the overfitting is observed or the metric
        doesn't improves for a predetermined number of epochs.
        '''

        should_exit = False
        is_candidate_for_best = False
        current_metric = round(accuracy, 4)
        # if current metric less than an average
        if current_metric <= self.prev_smooth_accuracy:
            self.iter_to_wait += 1
            if self.iter_to_wait >= self.train_patience:
                print("LOG:: The training should be stopped due to no improvements for {} epochs".format(self.train_patience))
                should_exit = True
        else:
            self.ema_smooth(accuracy)
            self.iter_to_wait = 0

        if current_metric >= self.best_metric:
            self.best_metric = current_metric
            is_candidate_for_best = True

        return should_exit, is_candidate_for_best

    def ema_smooth(self, cur_metric, alpha=0.8):
        """Exponential smoothing with factor `alpha`.
        """
        assert 0 < alpha <= 1
        self.prev_smooth_accuracy = alpha * cur_metric + (1. - alpha) * self.prev_smooth_accuracy
