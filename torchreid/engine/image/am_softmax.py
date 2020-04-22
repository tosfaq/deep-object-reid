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
import torch.nn as nn
import numpy as np

from torchreid.engine import Engine
from torchreid.utils import AverageMeter, open_specified_layers, open_all_layers
from torchreid.losses import get_regularizer, MetricLosses, AMSoftmaxLoss, CrossEntropyLoss, TotalVarianceLoss


class ImageAMSoftmaxEngine(Engine):
    r"""AM-Softmax-loss engine for image-reid.
    """

    def __init__(self, datamanager, model, optimizer, reg_cfg, metric_cfg,
                 scheduler=None, use_gpu=False, softmax_type='stock',
                 label_smooth=False, conf_penalty=False, pr_product=False,
                 m=0.35, s=10, end_s=None, duration_s=None, skip_steps_s=None, writer=None):
        super(ImageAMSoftmaxEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu)

        assert softmax_type in ['stock', 'am']
        assert s > 0.0
        if softmax_type == 'am':
            assert m >= 0.0

        self.regularizer = get_regularizer(reg_cfg)
        self.writer = writer
        self.enable_metric_losses = metric_cfg.enable

        num_batches = len(self.train_loader)
        num_classes = self.datamanager.num_train_pids
        if not isinstance(num_classes, (list, tuple)):
            num_classes = [num_classes]
        self.num_classes = num_classes
        self.num_targets = len(self.num_classes)

        self.main_losses = nn.ModuleList()
        for trg_num_classes in self.num_classes:
            scale_factor = np.log(trg_num_classes - 1) / np.log(self.num_classes[0] - 1)
            if softmax_type == 'stock':
                self.main_losses.append(CrossEntropyLoss(
                    use_gpu=self.use_gpu,
                    label_smooth=label_smooth,
                    conf_penalty=conf_penalty,
                    scale=scale_factor * s
                ))
            elif softmax_type == 'am':
                self.main_losses.append(AMSoftmaxLoss(
                    use_gpu=self.use_gpu,
                    label_smooth=label_smooth,
                    conf_penalty=conf_penalty,
                    m=m,
                    s=scale_factor * s,
                    end_s=scale_factor * end_s if self._valid(end_s) else None,
                    duration_s=duration_s * num_batches if self._valid(duration_s) else None,
                    skip_steps_s=skip_steps_s * num_batches if self._valid(skip_steps_s) else None,
                    pr_product=pr_product
                ))

        self.ml_loss = None
        if self.enable_metric_losses:
            self.ml_loss = MetricLosses(
                self.writer,
                num_classes[0],
                self.model.module.feature_dim,
                metric_cfg.center_coeff,
                metric_cfg.triplet_coeff,
            )

        # self.att_loss = TotalVarianceLoss(3, 1)
        self.att_loss = None

    @staticmethod
    def _valid(value):
        return value is not None and value > 0

    def train(self, epoch, max_epoch, writer, print_freq=10, fixbase_epoch=0, open_layers=None):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        reg_losses = AverageMeter()
        total_losses = AverageMeter()
        ml_losses = AverageMeter()
        att_losses = AverageMeter()
        main_losses = [AverageMeter() for _ in range(self.num_targets)]

        self.model.train()
        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch+1, fixbase_epoch))
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        num_batches = len(self.train_loader)
        start_time = time.time()
        for batch_idx, data in enumerate(self.train_loader):
            n_iter = epoch * num_batches + batch_idx
            data_time.update(time.time() - start_time)

            imgs, pids, trg_ids = self._parse_data_for_train(data, self.use_gpu)

            self.optimizer.zero_grad()
            all_logits, all_embeddings, glob_att = self.model(imgs, get_embeddings=True)

            total_loss = 0
            num_trg_losses = 0
            for trg_id in range(self.num_targets):
                trg_mask = trg_ids == trg_id

                trg_pids = pids[trg_mask]
                trg_num_samples = trg_pids.numel()
                if trg_num_samples == 0:
                    continue

                trg_logits = all_logits[trg_id][trg_mask]
                main_loss = self.main_losses[trg_id](trg_logits, trg_pids, iteration=n_iter)
                main_losses[trg_id].update(main_loss.item(), trg_pids.size(0))

                # if trg_id > 0:
                #     anchor_loss_avg = main_losses[0].avg
                #     ref_loss_avg = main_losses[trg_id].avg
                #     main_loss_scale = (anchor_loss_avg if anchor_loss_avg > 0.0 else 1.0) / \
                #                       (ref_loss_avg if ref_loss_avg > 0.0 else 1.0)
                #     main_loss *= main_loss_scale

                trg_loss = main_loss
                if trg_id == 0 and self.enable_metric_losses:
                    embd = all_embeddings[trg_id][trg_mask]
                    self.ml_loss.writer = writer
                    self.ml_loss.init_iteration()
                    ml_loss = self.ml_loss(embd, trg_pids, n_iter)
                    self.ml_loss.end_iteration()

                    ml_losses.update(ml_loss.item(), trg_pids.numel())
                    trg_loss += ml_loss

                total_loss += trg_loss
                num_trg_losses += 1

            total_loss /= float(num_trg_losses)

            if self.att_loss is not None:
                att_loss_val = 0.0
                for att_map in glob_att:
                    if att_map is not None:
                        att_loss_val += self.att_loss(att_map)

                if att_loss_val > 0.0:
                    att_losses.update(att_loss_val.item(), pids.size(0))
                    total_loss += att_loss_val

            if self.regularizer is not None and (epoch + 1) > fixbase_epoch:
                reg_loss = self.regularizer(self.model)
                reg_losses.update(reg_loss.item(), pids.size(0))

                total_loss += reg_loss

            total_loss.backward()
            self.optimizer.step()

            total_losses.update(total_loss.item(), pids.size(0))
            batch_time.update(time.time() - start_time)

            if print_freq > 0 and (batch_idx + 1) % print_freq == 0:
                eta_seconds = batch_time.avg * (num_batches - (batch_idx + 1) + (max_epoch - (epoch + 1)) * num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}] '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                      'Loss {loss.val:.3f} ({loss.avg:.3f}) '
                      'ML Loss {ml_loss.val:.3f} ({ml_loss.avg:.3f}) '
                      'Att Loss {att_loss.val:.3f} ({att_loss.avg:.3f}) '
                      'Lr {lr:.6f} '
                      'ETA {eta}'.
                      format(
                          epoch + 1, max_epoch, batch_idx + 1, num_batches,
                          batch_time=batch_time,
                          data_time=data_time,
                          loss=total_losses,
                          ml_loss=ml_losses,
                          att_loss=att_losses,
                          lr=self.optimizer.param_groups[0]['lr'],
                          eta=eta_str,
                      )
                )

                if writer is not None:
                    writer.add_scalar('Train/Time', batch_time.avg, n_iter)
                    writer.add_scalar('Train/Data', data_time.avg, n_iter)
                    info = self.main_losses[0].get_last_info()
                    for k in info:
                        writer.add_scalar('AUX info/' + k, info[k], n_iter)
                    writer.add_scalar('Loss/train', total_losses.avg, n_iter)
                    if (epoch + 1) > fixbase_epoch:
                        writer.add_scalar('Loss/reg_ow', reg_losses.avg, n_iter)
                    writer.add_scalar('Loss/att', att_losses.avg, n_iter)
                    writer.add_scalar('Aux/Learning_rate', self.optimizer.param_groups[0]['lr'], n_iter)
                    writer.add_scalar('Aux/Scale_main', self.main_losses[0].get_last_scale(), n_iter)
                    writer.add_scalar('Loss/ml', ml_losses.avg, n_iter)
                    for trg_id in range(self.num_targets):
                        writer.add_scalar('Loss/main_{}'.format(trg_id), main_losses[trg_id].avg, n_iter)
            start_time = time.time()

        if self.scheduler is not None:
            self.scheduler.step()

    @staticmethod
    def _parse_data_for_train(data, use_gpu):
        imgs = data[0]
        pids = data[1]
        dataset_id = data[4]

        if use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()
            dataset_id = dataset_id.cuda()

        return imgs, pids, dataset_id
