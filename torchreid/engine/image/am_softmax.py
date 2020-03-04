"""
 MIT License

 Copyright (c) 2018 Kaiyang Zhou

 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import datetime
import time

import torch

from torchreid import metrics
from torchreid.engine.image.softmax import ImageSoftmaxEngine
from torchreid.utils import AverageMeter, open_specified_layers, open_all_layers
from torchreid.losses import get_regularizer, MetricLosses, AMSoftmaxLoss, CrossEntropyLoss, MinEntropyLoss


class ImageAMSoftmaxEngine(ImageSoftmaxEngine):
    r"""AM-Softmax-loss engine for image-reid.
    """

    def __init__(self, datamanager, model, optimizer, reg_cfg, metric_cfg, extra_losses_cfg, batch_transform_cfg,
                 scheduler=None, use_gpu=False, softmax_type='stock', label_smooth=True, conf_penalty=False,
                 m=0.35, s=10, writer=None):
        super(ImageAMSoftmaxEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu)

        self.regularizer = get_regularizer(reg_cfg)
        self.writer = writer

        if softmax_type == 'stock':
            self.main_loss = CrossEntropyLoss(
                use_gpu=self.use_gpu,
                label_smooth=label_smooth,
                conf_penalty=conf_penalty
            )
        elif softmax_type == 'am':
            self.main_loss = AMSoftmaxLoss(
                use_gpu=self.use_gpu,
                conf_penalty=conf_penalty,
                m=m,
                s=s
            )
        else:
            raise ValueError('Unknown softmax type: {}'.format(softmax_type))

        self.batch_transform_cfg = batch_transform_cfg
        self.lambd_distr = torch.distributions.beta.Beta(self.batch_transform_cfg.alpha,
                                                         self.batch_transform_cfg.alpha)

        if metric_cfg.enable:
            num_classes = self.datamanager.num_train_pids
            if isinstance(num_classes, (list, tuple)):
                num_classes = num_classes[0]

            self.metric_losses = MetricLosses(num_classes,
                                              self.model.module.feature_dim, self.writer,
                                              metric_cfg.balance_losses,
                                              metric_cfg.center_coeff,
                                              metric_cfg.glob_push_plus_loss_coeff)
        else:
            self.metric_losses = None

        if extra_losses_cfg.enable:
            self.extra_tasks = extra_losses_cfg.tasks
            assert len(self.extra_tasks) > 0

            self.extra_pos_loss = AMSoftmaxLoss(
                use_gpu=self.use_gpu,
                conf_penalty=extra_losses_cfg.conf_penalty,
                m=extra_losses_cfg.m,
                s=extra_losses_cfg.s
            )
            self.extra_neg_loss = MinEntropyLoss(scale=extra_losses_cfg.s)
        else:
            self.extra_pos_loss = None
            self.extra_tasks = None

    def train(self, epoch, max_epoch, writer, print_freq=10, fixbase_epoch=0, open_layers=None):
        losses = AverageMeter()
        real_loss = AverageMeter()
        synthetic_loss = AverageMeter()
        reg_ow_loss = AverageMeter()
        metric_losses = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        attr_loss = AverageMeter()
        if self.extra_tasks is not None:
            attr_pos_losses = dict()
            attr_neg_losses = dict()
            for attr_loss_name in self.extra_tasks.keys():
                attr_pos_losses[attr_loss_name] = AverageMeter()
                attr_neg_losses[attr_loss_name] = AverageMeter()

        self.model.train()
        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch+1, fixbase_epoch))
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        num_batches = len(self.train_loader)
        start_time = time.time()
        for batch_idx, data in enumerate(self.train_loader):
            data_time.update(time.time() - start_time)

            imgs, pids = self._parse_data_for_train(data)
            imgs = self._apply_batch_transform(imgs)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()

            batch_size = pids.size(0)

            self.optimizer.zero_grad()
            if self.metric_losses is not None:
                embeddings, outputs, extra_outputs = self.model(imgs, get_embeddings=True)
            else:
                outputs, extra_outputs = self.model(imgs)
                embeddings = None

            if isinstance(outputs, dict):
                real_data_mask = self._get_real_mask(data, self.use_gpu)
                synthetic_data_mask = ~real_data_mask

                real_outputs = outputs['real'][real_data_mask]
                synthetic_outputs = outputs['synthetic'][synthetic_data_mask]

                real_pids = pids[real_data_mask]
                synthetic_pids = pids[synthetic_data_mask]

                real_embeddings = None
                if embeddings is not None:
                    real_embeddings = embeddings['real'][real_data_mask]

                loss = torch.zeros([], dtype=real_outputs.dtype, device=real_outputs.device)
                num_losses = 0
                if real_pids.numel() > 0:
                    real_data_loss = self._compute_loss(self.main_loss, real_outputs, real_pids)
                    real_loss.update(real_data_loss.item(), batch_size)
                    loss += real_data_loss
                    num_losses += 1
                if synthetic_pids.numel() > 0:
                    synthetic_data_loss = self._compute_loss(self.main_loss, synthetic_outputs, synthetic_pids)
                    synthetic_loss.update(synthetic_data_loss.item(), batch_size)
                    num_losses += 1
                    loss += synthetic_data_loss
                loss /= num_losses
            else:
                real_outputs = outputs
                real_pids = pids
                real_embeddings = embeddings

                loss = self._compute_loss(self.main_loss, real_outputs, real_pids)
                real_loss.update(loss.item(), batch_size)

            if self.extra_tasks is not None:
                extra_labels = self._parse_extra_data_for_train(data, self.use_gpu)
                attr_losses_list = []
                for task_name in self.extra_tasks.keys():
                    task_labels = extra_labels[task_name]
                    task_outputs = extra_outputs[task_name]

                    pos_mask = task_labels >= 0
                    pos_labels = task_labels[pos_mask]
                    if pos_labels.numel() > 0:
                        pos_outputs = task_outputs[pos_mask]
                        extra_pos_loss = self.extra_pos_loss(pos_outputs, pos_labels)
                    else:
                        extra_pos_loss = torch.zeros([], dtype=loss.dtype, device=loss.device)
                    attr_pos_losses[task_name].update(extra_pos_loss.item(), batch_size)

                    neg_mask = task_labels < 0
                    neg_outputs = task_outputs[neg_mask]
                    extra_neg_loss = self.extra_neg_loss(neg_outputs)
                    attr_neg_losses[task_name].update(extra_neg_loss.item(), batch_size)

                    extra_loss_value = extra_pos_loss + extra_neg_loss
                    attr_losses_list.append(extra_loss_value)

                attr_loss_value = torch.stack(attr_losses_list).sum()
                loss += attr_loss_value
                attr_loss.update(attr_loss_value.item(), batch_size)

            if (epoch + 1) > fixbase_epoch:
                reg_loss = self.regularizer(self.model)
                reg_ow_loss.update(reg_loss.item(), batch_size)
                loss += reg_loss

            if self.metric_losses is not None and real_embeddings is not None:
                if real_pids.numel() > 0:
                    self.metric_losses.writer = writer
                    self.metric_losses.init_iteration()
                    metric_loss = self.metric_losses(real_embeddings, real_pids,
                                                     epoch, epoch * num_batches + batch_idx)
                    self.metric_losses.end_iteration()
                else:
                    metric_loss = torch.zeros([], dtype=loss.dtype, device=loss.device)
                loss += metric_loss
                metric_losses.update(metric_loss.item(), batch_size)

            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), batch_size)
            accs.update(metrics.accuracy(real_outputs, real_pids)[0].item())
            batch_time.update(time.time() - start_time)

            if print_freq > 0 and (batch_idx + 1) % print_freq == 0:
                eta_seconds = batch_time.avg * (num_batches-(batch_idx + 1) + (max_epoch - (epoch + 1)) * num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}] '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                      'ML Loss {ml_loss.val:.4f} ({ml_loss.avg:.4f}) '
                      'Attr Loss {attr_loss.val:.4f} ({attr_loss.avg:.4f}) '
                      'Acc {acc.val:.2f} ({acc.avg:.2f}) '
                      'Lr {lr:.6f} '
                      'eta {eta}'.
                      format(
                          epoch + 1, max_epoch, batch_idx + 1, num_batches,
                          batch_time=batch_time,
                          data_time=data_time,
                          ml_loss=metric_losses,
                          attr_loss=attr_loss,
                          loss=losses,
                          acc=accs,
                          lr=self.optimizer.param_groups[0]['lr'],
                          eta=eta_str,
                      )
                )

                if writer is not None:
                    n_iter = epoch * num_batches + batch_idx
                    writer.add_scalar('Train/Time', batch_time.avg, n_iter)
                    writer.add_scalar('Train/Data', data_time.avg, n_iter)
                    info = self.main_loss.get_last_info()
                    for k in info:
                        writer.add_scalar('AUX info/' + k, info[k], n_iter)
                    writer.add_scalar('Loss/train', losses.avg, n_iter)
                    writer.add_scalar('Loss/train_real', real_loss.avg, n_iter)
                    writer.add_scalar('Loss/train_synth', synthetic_loss.avg, n_iter)
                    if (epoch + 1) > fixbase_epoch:
                        writer.add_scalar('Loss/reg_ow', reg_ow_loss.avg, n_iter)
                    writer.add_scalar('Accuracy/train', accs.avg, n_iter)
                    writer.add_scalar('Learning rate', self.optimizer.param_groups[0]['lr'], n_iter)
                    if self.extra_tasks is not None:
                        for task_name in self.extra_tasks.keys():
                            writer.add_scalar('Loss/pos_{}'.format(task_name),
                                              attr_pos_losses[task_name].avg, n_iter)
                            writer.add_scalar('Loss/neg_{}'.format(task_name),
                                              attr_neg_losses[task_name].avg, n_iter)
            start_time = time.time()

        if self.scheduler is not None:
            self.scheduler.step()

    @staticmethod
    def _parse_extra_data_for_train(data, use_gpu=False):
        return dict(attr_color=data[4].cuda() if use_gpu else data[4],
                    attr_type=data[5].cuda() if use_gpu else data[5])

    @staticmethod
    def _get_real_mask(data, use_gpu=False):
        mask = data[4] < 0
        if use_gpu:
            mask = mask.cuda()

        return mask

    def _apply_batch_transform(self, imgs):
        if self.batch_transform_cfg.enable:
            permuted_idx = torch.randperm(imgs.shape[0])
            lambd = self.batch_transform_cfg.anchor_bias \
                    + (1 - self.batch_transform_cfg.anchor_bias) \
                    * self.lambd_distr.sample((imgs.shape[0],))
            imgs = lambd * imgs + (1 - lambd) * imgs[permuted_idx]

        return imgs
