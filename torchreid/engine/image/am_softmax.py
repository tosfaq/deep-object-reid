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
import numpy as np

from torchreid import metrics
from torchreid.engine.image.softmax import ImageSoftmaxEngine
from torchreid.utils import AverageMeter, open_specified_layers, open_all_layers
from torchreid.losses import get_regularizer, MetricLosses, AMSoftmaxLoss, CrossEntropyLoss, MinEntropyLoss, set_kl_div


class ImageAMSoftmaxEngine(ImageSoftmaxEngine):
    r"""AM-Softmax-loss engine for image-reid.
    """

    def __init__(self, datamanager, model, optimizer, reg_cfg, metric_cfg, attr_losses_cfg, batch_transform_cfg,
                 scheduler=None, use_gpu=False, softmax_type='stock', label_smooth=False, conf_penalty=False,
                 m=0.35, s=10, end_s=None, duration_s=None, skip_steps_s=None, writer=None):
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
            num_batches = len(self.train_loader)
            self.main_loss = AMSoftmaxLoss(
                use_gpu=self.use_gpu,
                label_smooth=label_smooth,
                conf_penalty=conf_penalty,
                m=m,
                s=s,
                end_s=end_s,
                duration_s=duration_s * num_batches if self._valid(duration_s) else None,
                skip_steps_s=skip_steps_s * num_batches if self._valid(skip_steps_s) else None
            )
        else:
            raise ValueError('Unknown softmax type: {}'.format(softmax_type))

        self.batch_transform_cfg = batch_transform_cfg
        self.lambd_distr = torch.distributions.beta.Beta(self.batch_transform_cfg.alpha,
                                                         self.batch_transform_cfg.alpha)

        self.enable_metric_losses = metric_cfg.enable
        if self.enable_metric_losses:
            self.real_metric_loss = MetricLosses(
                self.writer, metric_cfg.center_coeff, metric_cfg.glob_push_coeff,
                metric_cfg.local_push_coeff, metric_cfg.pull_coeff)
            self.synthetic_metric_loss = None
            if self.model.module.split_embeddings:
                self.synthetic_metric_loss = MetricLosses(
                    self.writer, metric_cfg.center_coeff, metric_cfg.glob_push_coeff,
                    metric_cfg.local_push_coeff, metric_cfg.pull_coeff)

        if attr_losses_cfg.enable:
            self.attr_tasks = attr_losses_cfg.tasks
            assert len(self.attr_tasks) > 0

            num_batches = len(self.train_loader)
            self.attr_pos_loss = AMSoftmaxLoss(
                use_gpu=self.use_gpu,
                conf_penalty=attr_losses_cfg.conf_penalty,
                label_smooth=attr_losses_cfg.label_smooth,
                m=attr_losses_cfg.m,
                s=attr_losses_cfg.s,
                end_s=attr_losses_cfg.end_s,
                duration_s=attr_losses_cfg.duration_s * num_batches if self._valid(attr_losses_cfg.duration_s) else None,
                skip_steps_s=attr_losses_cfg.skip_steps_s * num_batches if self._valid(attr_losses_cfg.skip_steps_s) else None
            )
            self.attr_neg_loss = MinEntropyLoss(scale=attr_losses_cfg.s)
            self.attr_neg_scale = 6.0
            self.attr_lr = 0.01
            # self.attr_factors = dict()
            # for task_name in self.attr_tasks:
            #     self.attr_factors[task_name] = 0.5
        else:
            self.attr_pos_loss = None
            self.attr_tasks = None

    @staticmethod
    def _valid(value):
        return value is not None and value > 0

    def train(self, epoch, max_epoch, writer, print_freq=10, fixbase_epoch=0, open_layers=None):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        accs = AverageMeter(enable_zeros=True)
        losses = AverageMeter()
        trg_losses = AverageMeter()
        real_loss = AverageMeter()
        synthetic_loss = AverageMeter()
        reg_ow_loss = AverageMeter()
        metric_losses = AverageMeter()
        attr_loss = AverageMeter()
        if self.attr_tasks is not None:
            attr_pos_losses = dict()
            attr_neg_losses = dict()
            attr_neg_sync_losses = dict()
            for attr_loss_name in self.attr_tasks.keys():
                attr_pos_losses[attr_loss_name] = AverageMeter()
                attr_neg_losses[attr_loss_name] = AverageMeter()
                attr_neg_sync_losses[attr_loss_name] = AverageMeter()

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

            imgs, pids, cam_ids = self._parse_data_for_train(data)
            imgs = self._apply_batch_transform(imgs)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()
                cam_ids = cam_ids.cuda()

            batch_size = pids.size(0)

            self.optimizer.zero_grad()
            if self.enable_metric_losses:
                embeddings, outputs, extra_outputs = self.model(imgs, get_embeddings=True)
            else:
                outputs, extra_outputs = self.model(imgs)
                num_parts = self.model.module.num_parts + 1
                embeddings = dict(real=[None] * num_parts, synthetic=[None] * num_parts)

            real_centers = outputs['real_centers']
            synthetic_centers = outputs['synthetic_centers']

            if self.model.module.split_embeddings:
                real_mask = self._get_real_mask(data, self.use_gpu)
                synthetic_mask = ~real_mask

                real_outputs = [out[real_mask] for out in outputs['real']]
                synthetic_outputs = [out[synthetic_mask] for out in outputs['synthetic']]

                real_pids = pids[real_mask]
                synthetic_pids = pids[synthetic_mask]

                real_cam_ids = cam_ids[real_mask]
                synthetic_cam_ids = cam_ids[synthetic_mask]

                real_embeddings = [e[real_mask] if e is not None else None for e in embeddings['real']]
                synthetic_embeddings = [e[synthetic_mask] if e is not None else None for e in embeddings['synthetic']]

                trg_loss = torch.zeros([], dtype=real_outputs[0].dtype, device=real_outputs[0].device)
                num_losses = 0
                if real_pids.numel() > 0:
                    real_data_loss = self._compute_loss(self.main_loss, real_outputs, real_pids, iteration=n_iter)
                    real_loss.update(real_data_loss.item(), real_pids.numel())
                    trg_loss += real_data_loss
                    num_losses += 1
                if synthetic_pids.numel() > 0:
                    synthetic_data_loss =\
                        self._compute_loss(self.main_loss, synthetic_outputs, synthetic_pids, iteration=n_iter)
                    synthetic_loss.update(synthetic_data_loss.item(), synthetic_pids.numel())

                    synthetic_loss_weight = (real_loss.avg if real_loss.avg > 0.0 else 1.0) / \
                                            (synthetic_loss.avg if synthetic_loss.avg > 0.0 else 1.0)
                    trg_loss += synthetic_loss_weight * synthetic_data_loss
                    num_losses += 1
                trg_loss /= num_losses
            else:
                real_outputs = outputs['real']
                real_pids = pids
                real_cam_ids = cam_ids
                real_embeddings = embeddings['real']
                synthetic_embeddings = embeddings['synthetic']

                trg_loss = self._compute_loss(self.main_loss, real_outputs, real_pids, iteration=n_iter)
                real_loss.update(trg_loss.item(), batch_size)
            trg_losses.update(trg_loss.item(), batch_size)

            if self.attr_tasks is not None:
                attr_labels_dict = self._parse_attr_data_for_train(data, self.use_gpu)
                attr_losses_list = []
                for attr_name in self.attr_tasks.keys():
                    attr_labels = attr_labels_dict[attr_name]
                    attr_outputs = extra_outputs[attr_name]

                    pos_mask = attr_labels >= 0
                    pos_labels = attr_labels[pos_mask]
                    if pos_mask.numel() > 0:
                        pos_outputs = attr_outputs[pos_mask]
                        attr_pos_loss = self.attr_pos_loss(pos_outputs, pos_labels, iteration=n_iter)
                    else:
                        attr_pos_loss = torch.zeros([], dtype=trg_loss.dtype, device=trg_loss.device)
                    attr_pos_losses[attr_name].update(attr_pos_loss.item(), pos_mask.numel())

                    neg_mask = attr_labels < 0
                    neg_outputs = attr_outputs[neg_mask]
                    attr_neg_loss = self.attr_neg_loss(neg_outputs, scale=self.attr_pos_loss.get_last_scale())
                    attr_neg_losses[attr_name].update(attr_neg_loss.item(), neg_mask.numel())

                    attr_neg_sync_loss = torch.zeros([], dtype=trg_loss.dtype, device=trg_loss.device)
                    unique_neg_pids = np.unique(real_pids.data.cpu().numpy())
                    for unique_neg_pid in unique_neg_pids:
                        neg_pid_outputs = neg_outputs[real_pids == unique_neg_pid]
                        attr_neg_sync_loss += set_kl_div(neg_pid_outputs * self.attr_pos_loss.get_last_scale())
                    attr_neg_sync_loss /= float(max(1, len(unique_neg_pids)))
                    attr_neg_sync_losses[attr_name].update(attr_neg_sync_loss.item(), len(unique_neg_pids))

                    # attr_pos_scalar = attr_pos_loss.item()
                    # att_neg_scalar = attr_neg_loss.item() + attr_neg_sync_loss.item()
                    # attr_diff_scalar = self.attr_neg_scale * att_neg_scalar - attr_pos_scalar
                    # attr_factor = np.clip(self.attr_factors[attr_name] + self.attr_lr * attr_diff_scalar, 0.0, 1.0)
                    # # attr_total_loss = attr_factor * self.attr_neg_scale * (attr_neg_loss + attr_neg_sync_loss) + \
                    # #                   (1.0 - attr_factor) * attr_pos_loss
                    attr_total_loss = attr_pos_loss + attr_neg_sync_loss
                    attr_losses_list.append(attr_total_loss)
                    # self.attr_factors[attr_name] = attr_factor

                attr_loss_value = torch.stack(attr_losses_list).mean()
                attr_loss.update(attr_loss_value.item(), batch_size)

                # attr_loss_weight = (trg_losses.avg if trg_losses.avg > 0.0 else 1.0) / \
                #                    (attr_loss.avg if attr_loss.avg > 0.0 else 1.0)
                # total_loss = 0.5 * (trg_loss + attr_loss_weight * attr_loss_value)
                total_loss = trg_loss + attr_loss_value
            else:
                total_loss = trg_loss

            if (epoch + 1) > fixbase_epoch:
                reg_loss = self.regularizer(self.model)
                reg_ow_loss.update(reg_loss.item(), batch_size)
                total_loss += reg_loss

            if self.enable_metric_losses:
                num_real_embeddings = sum([True for e in real_embeddings if e is not None])
                real_metric_loss = torch.zeros([], dtype=total_loss.dtype, device=total_loss.device)
                if num_real_embeddings > 0 and real_pids.numel() > 0:
                    for embd_id, embd in enumerate(real_embeddings):
                        if embd is None:
                            continue

                        self.real_metric_loss.writer = writer
                        name = 'real_{}'.format(embd_id)
                        real_metric_loss += self.real_metric_loss(
                            embd, real_centers[embd_id], real_pids, real_cam_ids, n_iter, name)
                    real_metric_loss /= float(num_real_embeddings)

                num_synthetic_embeddings = sum([True for e in synthetic_embeddings if e is not None])
                synthetic_metric_loss = torch.zeros([], dtype=total_loss.dtype, device=total_loss.device)
                if self.model.module.split_embeddings and num_synthetic_embeddings > 0 and synthetic_pids.numel() > 0:
                    for embd_id, embd in enumerate(synthetic_embeddings):
                        if embd is None:
                            continue

                        self.synthetic_metric_loss.writer = writer
                        name = 'synthetic_{}'.format(embd_id)
                        synthetic_metric_loss += self.synthetic_metric_loss(
                            embd, synthetic_centers[embd_id], synthetic_pids, synthetic_cam_ids, n_iter, name)
                    synthetic_metric_loss /= float(num_synthetic_embeddings)

                metric_loss = 0.5 * (real_metric_loss + synthetic_metric_loss)
                # metric_loss = real_metric_loss
                metric_losses.update(metric_loss.item(), batch_size)

                total_loss += metric_loss

            total_loss.backward()
            self.optimizer.step()

            losses.update(total_loss.item(), batch_size)
            accs.update(metrics.accuracy(real_outputs[0], real_pids)[0].item())
            batch_time.update(time.time() - start_time)

            if print_freq > 0 and (batch_idx + 1) % print_freq == 0:
                eta_seconds = batch_time.avg * (num_batches - (batch_idx + 1) + (max_epoch - (epoch + 1)) * num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}] '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                      'ML Loss {ml_loss.val:.4f} ({ml_loss.avg:.4f}) '
                      'Attr Loss {attr_loss.val:.4f} ({attr_loss.avg:.4f}) '
                      'Acc {acc.val:.2f} ({acc.avg:.2f}) '
                      'Lr {lr:.6f} '
                      'Scale {scale:.2f} '
                      'ETA {eta}'.
                      format(
                          epoch + 1, max_epoch, batch_idx + 1, num_batches,
                          batch_time=batch_time,
                          data_time=data_time,
                          ml_loss=metric_losses,
                          attr_loss=attr_loss,
                          loss=losses,
                          acc=accs,
                          lr=self.optimizer.param_groups[0]['lr'],
                          scale=self.main_loss.get_last_scale(),
                          eta=eta_str,
                      )
                )

                if writer is not None:
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
                    writer.add_scalar('Aux/Learning_rate', self.optimizer.param_groups[0]['lr'], n_iter)
                    writer.add_scalar('Aux/Scale_main', self.main_loss.get_last_scale(), n_iter)
                    if self.attr_pos_loss is not None:
                        writer.add_scalar('Aux/Scale_attr', self.attr_pos_loss.get_last_scale(), n_iter)
                    if self.attr_tasks is not None:
                        for attr_name in self.attr_tasks.keys():
                            writer.add_scalar('Loss/pos_{}'.format(attr_name),
                                              attr_pos_losses[attr_name].avg, n_iter)
                            writer.add_scalar('Loss/neg_{}'.format(attr_name),
                                              attr_neg_losses[attr_name].avg, n_iter)
                            writer.add_scalar('Loss/neg_sync_{}'.format(attr_name),
                                              attr_neg_sync_losses[attr_name].avg, n_iter)
                            # writer.add_scalar('Loss/factor_{}'.format(attr_name),
                            #                   self.attr_factors[attr_name], n_iter)
            start_time = time.time()

        if self.scheduler is not None:
            self.scheduler.step()

    @staticmethod
    def _parse_data_for_train(data):
        imgs = data[0]
        pids = data[1]
        cam_ids = data[2]
        return imgs, pids, cam_ids

    @staticmethod
    def _parse_attr_data_for_train(data, use_gpu=False):
        return dict(attr_color=data[4].cuda() if use_gpu else data[4],
                    attr_type=data[5].cuda() if use_gpu else data[5],
                    attr_orientation=data[6].cuda() if use_gpu else data[6])

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
