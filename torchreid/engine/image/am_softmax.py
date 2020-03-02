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
from torchreid.utils import AverageMeter, open_specified_layers, open_all_layers, visualize_ranked_results, re_ranking
from torchreid.losses.am_softmax import AMSoftmaxLoss
from torchreid.losses.cross_entropy_loss import CrossEntropyLoss
from torchreid.losses.regularizers import get_regularizer
from torchreid.losses.metric import MetricLosses


class ImageAMSoftmaxEngine(ImageSoftmaxEngine):
    r"""AM-Softmax-loss engine for image-reid.
    """

    def __init__(self, datamanager, model, optimizer, reg_cfg, metric_cfg, batch_transform_cfg,
                 scheduler=None, use_gpu=False, softmax_type='stock', label_smooth=True, conf_penalty=False,
                 m=0.35, s=10, writer=None):
        super(ImageAMSoftmaxEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu)

        self.regularizer = get_regularizer(reg_cfg)
        self.writer = writer

        if softmax_type == 'stock':
            self.criterion = CrossEntropyLoss(
                num_classes=self.datamanager.num_train_pids,
                use_gpu=self.use_gpu,
                label_smooth=label_smooth,
                conf_penalty=conf_penalty
            )
        elif softmax_type == 'am':
            self.criterion = AMSoftmaxLoss(
                num_classes=self.datamanager.num_train_pids,
                use_gpu=self.use_gpu,
                conf_penalty=conf_penalty,
                m=m,
                s=s
            )

        self.batch_transform_cfg = batch_transform_cfg
        self.lambd_distr = torch.distributions.beta.Beta(self.batch_transform_cfg.alpha,
                                                         self.batch_transform_cfg.alpha)

        if metric_cfg.enable:
            self.metric_losses = MetricLosses(self.datamanager.num_train_pids,
                                              self.model.module.feature_dim, self.writer,
                                              metric_cfg.balance_losses,
                                              metric_cfg.center_coeff,
                                              metric_cfg.glob_push_plus_loss_coeff)
        else:
            self.metric_losses = None

    def train(self, epoch, max_epoch, writer, print_freq=10, fixbase_epoch=0, open_layers=None):
        losses = AverageMeter()
        reg_ow_loss = AverageMeter()
        metric_losses = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

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

            self.optimizer.zero_grad()
            if self.metric_losses is not None:
                embeddings, outputs = self.model(imgs, get_embeddings=True)
            else:
                outputs = self.model(imgs)

            loss = self._compute_loss(self.criterion, outputs, pids)

            if (epoch + 1) > fixbase_epoch:
                reg_loss = self.regularizer(self.model)
                reg_ow_loss.update(reg_loss.item(), pids.size(0))
                loss += reg_loss

            if self.metric_losses is not None:
                self.metric_losses.writer = writer
                self.metric_losses.init_iteration()
                metric_loss = self.metric_losses(embeddings, pids, epoch, epoch * num_batches + batch_idx)
                self.metric_losses.end_iteration()
                loss += metric_loss
                metric_losses.update(metric_loss.item(), pids.size(0))

            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), pids.size(0))
            accs.update(metrics.accuracy(outputs, pids)[0].item())
            batch_time.update(time.time() - start_time)

            if print_freq > 0 and (batch_idx + 1) % print_freq == 0:
                eta_seconds = batch_time.avg * (num_batches-(batch_idx + 1) + (max_epoch - (epoch + 1)) * num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'AUX Losses {aux_losses.val:.4f} ({aux_losses.avg:.4f})\t'
                      'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                      'Lr {lr:.6f}\t'
                      'eta {eta}'.
                      format(
                          epoch + 1, max_epoch, batch_idx + 1, num_batches,
                          batch_time=batch_time,
                          data_time=data_time,
                          aux_losses=metric_losses,
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
                    info = self.criterion.get_last_info()
                    for k in info:
                        writer.add_scalar('AUX info/' + k, info[k], n_iter)
                    writer.add_scalar('Loss/train', losses.avg, n_iter)
                    if (epoch + 1) > fixbase_epoch:
                        writer.add_scalar('Loss/reg_ow', reg_ow_loss.avg, n_iter)
                    writer.add_scalar('Accuracy/train', accs.avg, n_iter)
                    writer.add_scalar('Learning rate', self.optimizer.param_groups[0]['lr'], n_iter)
            start_time = time.time()

        if self.scheduler is not None:
            self.scheduler.step()

    def _apply_batch_transform(self, imgs):
        if self.batch_transform_cfg.enable:
            permuted_idx = torch.randperm(imgs.shape[0])
            lambd = self.batch_transform_cfg.anchor_bias \
                    + (1 - self.batch_transform_cfg.anchor_bias) \
                    * self.lambd_distr.sample((imgs.shape[0],))
            imgs = lambd * imgs + (1 - lambd) * imgs[permuted_idx]

        return imgs
