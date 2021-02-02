"""
 Copyright (c) 2020 Intel Corporation

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

from __future__ import division, print_function, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchreid.engine import Engine
from torchreid.losses import InfoNCELoss, get_regularizer


class ImageContrastiveEngine(Engine):
    r"""InfoNCE-loss engine for image-classification.
    """

    def __init__(self, datamanager, model, optimizer, reg_cfg, scheduler=None, use_gpu=False,
                 s=10, end_s=None, duration_s=None, skip_steps_s=None):
        super(ImageContrastiveEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu)

        self.regularizer = get_regularizer(reg_cfg)

        num_batches = len(self.train_loader)
        num_classes = self.datamanager.num_train_pids
        if not isinstance(num_classes, (list, tuple)):
            num_classes = [num_classes]
        self.num_classes = num_classes
        self.num_targets = len(self.num_classes)

        self.main_losses = nn.ModuleList()
        for trg_id, _ in enumerate(self.num_classes):
            self.main_losses.append(InfoNCELoss(
                use_gpu=self.use_gpu,
                s=s,
                end_s=end_s if self._valid(end_s) else None,
                duration_s=duration_s * num_batches if self._valid(duration_s) else None,
                skip_steps_s=skip_steps_s * num_batches if self._valid(skip_steps_s) else None,
            ))

    @staticmethod
    def _valid(value):
        return value is not None and value > 0

    def forward_backward(self, data):
        n_iter = self.epoch * self.num_batches + self.batch_idx

        imgs, dataset_ids = self.parse_data_for_train(data, self.use_gpu)

        model_names = self.get_model_names()
        num_models = len(model_names)

        out_embd = [[] for _ in range(self.num_targets)]
        total_loss = torch.zeros([], dtype=imgs.dtype, device=imgs.device)
        loss_summary = dict()

        for model_name in model_names:
            self.optims[model_name].zero_grad()

            model_loss, model_loss_summary, model_embd = self._single_model_losses(
                self.models[model_name], imgs, dataset_ids, model_name, n_iter
            )

            total_loss += model_loss / float(num_models)
            loss_summary.update(model_loss_summary)

            for trg_id in range(self.num_targets):
                if model_embd[trg_id] is not None:
                    out_embd[trg_id].append(model_embd[trg_id])

        if len(model_names) > 1:
            num_mutual_losses = 0
            mutual_loss = torch.zeros([], dtype=imgs.dtype, device=imgs.device)
            for trg_id in range(self.num_targets):
                if len(out_embd[trg_id]) <= 1:
                    continue

                with torch.no_grad():
                    trg_embd = torch.stack(out_embd[trg_id]).mean(dim=0)
                    trg_norm_embd = F.normalize(trg_embd, p=2, dim=1)

                for model_id, embd in enumerate(out_embd[trg_id]):
                    norm_embd = F.normalize(embd, p=2, dim=1)
                    m_losses = 1.0 - (trg_norm_embd * norm_embd).sum(dim=1)
                    m_loss = m_losses.mean()

                    mutual_loss += m_loss
                    loss_summary['mutual_{}/{}'.format(trg_id, model_names[model_id])] = m_loss.item()
                    num_mutual_losses += 1

            total_loss += mutual_loss / float(num_mutual_losses)

        total_loss.backward()

        for model_name in model_names:
            self.optims[model_name].step()

        loss_summary['loss'] = total_loss.item()

        return loss_summary, 0.0

    def _single_model_losses(self, model, imgs, dataset_ids, model_name, n_iter):
        all_embeddings = model(imgs)

        total_loss = torch.zeros([], dtype=imgs.dtype, device=imgs.device)
        out_embd = []
        loss_summary = dict()

        num_trg_losses = 0
        for trg_id in range(self.num_targets):
            trg_mask = dataset_ids == trg_id

            trg_num_samples = trg_mask.sum()
            if trg_num_samples == 0:
                out_embd.append(None)
                continue

            embd = all_embeddings[trg_id][trg_mask]
            out_embd.append(embd)

            main_loss = self.main_losses[trg_id](embd, iteration=n_iter)
            loss_summary['main_{}/{}'.format(trg_id, model_name)] = main_loss.item()

            total_loss += main_loss
            num_trg_losses += 1
        total_loss /= float(num_trg_losses)

        if self.regularizer is not None and (self.epoch + 1) > self.fixbase_epoch:
            reg_loss = self.regularizer(model)

            loss_summary['reg/{}'.format(model_name)] = reg_loss.item()
            total_loss += reg_loss

        return total_loss, loss_summary, out_embd

    @staticmethod
    def parse_data_for_train(data, use_gpu=False):
        imgs = data[0]
        dataset_ids = data[3]
        assert len(imgs.size()) == 5

        if use_gpu:
            imgs = imgs.cuda()
            dataset_ids = dataset_ids.cuda()

        b, num_packages, c, h, w = imgs.size()
        assert num_packages == 2

        imgs = imgs.view(b * num_packages, c, h, w)
        dataset_ids = dataset_ids.view(-1, 1).repeat(1, num_packages).view(-1)

        return imgs, dataset_ids
