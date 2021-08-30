from __future__ import absolute_import, division, print_function

import torch

from torchreid import metrics
from torchreid.losses import CrossEntropyLoss
from ..engine import Engine


class ImageSoftmaxEngine(Engine):
    r"""Softmax-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        scheduler=None,
        use_gpu=True,
        label_smooth=True,
        conf_penalty=0.0
    ):
        super(ImageSoftmaxEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu)

        self.main_real_loss = CrossEntropyLoss(
            use_gpu=self.use_gpu,
            label_smooth=label_smooth,
            conf_penalty=conf_penalty
        )

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)
        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        model_names = self.get_model_names()
        num_models = len(model_names)

        all_losses, all_logits = [], []
        loss_summary = dict()

        for model_name in model_names:
            self.optims[model_name].zero_grad()

            logits = self.models[model_name](imgs)
            all_logits.append(logits)

            loss = self.compute_loss(self.main_real_loss, logits, pids)
            all_losses.append(loss / float(num_models))

            loss_summary['{}/main'.format(model_name)] = loss.item()
            loss_summary['{}/acc'.format(model_name)] = metrics.accuracy(logits, pids)[0].item()

        if len(all_logits) > 1:
            with torch.no_grad():
                trg_probs = torch.softmax(torch.stack(all_logits), dim=2).mean(dim=0)

            mix_loss = 0.0
            for logits in all_logits:
                log_probs = torch.log_softmax(logits, dim=1)
                mix_loss += (trg_probs * log_probs).mean().neg()
            mix_loss /= float(len(all_logits))

            all_losses.append(mix_loss)
            loss_summary['mix'] = mix_loss.item()

        total_loss = sum(all_losses)
        total_loss.backward()

        for model_name in model_names:
            self.optims[model_name].step()

        loss_summary['loss'] = total_loss.item()

        return loss_summary
