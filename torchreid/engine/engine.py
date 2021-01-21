from __future__ import division, print_function, absolute_import
import time
import numpy as np
import os
import os.path as osp
import datetime
from collections import OrderedDict
import torch
import copy
from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from torchreid import metrics
from torchreid.utils import (
    MetricMeter, AverageMeter, re_ranking, open_all_layers, save_checkpoint,
    open_specified_layers, visualize_ranked_results, get_model_attr
)
from torchreid.losses import DeepSupervision


class Engine:
    r"""A generic base Engine class for both image- and video-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(self, datamanager, models, optimizers, schedulers, use_gpu=True, save_chkpt=True):
        self.datamanager = datamanager
        self.train_loader = self.datamanager.train_loader
        self.test_loader = self.datamanager.test_loader
        self.use_gpu = (torch.cuda.is_available() and use_gpu)
        self.save_chkpt = save_chkpt
        self.writer = None

        self.start_epoch = 0
        self.fixbase_epoch = 0
        self.max_epoch = None
        self.num_batches = None
        self.epoch = None

        self.models = OrderedDict()
        self.optims = OrderedDict()
        self.scheds = OrderedDict()

        if isinstance(models, (tuple, list)):
            assert isinstance(optimizers, (tuple, list))
            assert isinstance(schedulers, (tuple, list))

            num_models = len(models)
            assert len(optimizers) == num_models
            assert len(schedulers) == num_models

            for model_id, (model, optimizer, scheduler) in enumerate(zip(models, optimizers, schedulers)):
                self.register_model(f'model_{model_id}', model, optimizer, scheduler)
        else:
            assert not isinstance(optimizers, (tuple, list))
            assert not isinstance(schedulers, (tuple, list))

            self.register_model('model', models, optimizers, schedulers)

    def register_model(self, name='model', model=None, optim=None, sched=None):
        if self.__dict__.get('models') is None:
            raise AttributeError(
                'Cannot assign model before super().__init__() call'
            )

        if self.__dict__.get('optims') is None:
            raise AttributeError(
                'Cannot assign optim before super().__init__() call'
            )

        if self.__dict__.get('scheds') is None:
            raise AttributeError(
                'Cannot assign sched before super().__init__() call'
            )

        self.models[name] = model
        self.optims[name] = optim
        self.scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self.models.keys())
        if names is not None:
            if not isinstance(names, list):
                names = [names]
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(self, epoch, save_dir, is_best=False):
        names = self.get_model_names()

        for name in names:
            ckpt_name = osp.join(save_dir, name)
            save_checkpoint(
                {
                    'state_dict': self.models[name].state_dict(),
                    'epoch': epoch + 1,
                    'optimizer': self.optims[name].state_dict(),
                    'scheduler': self.scheds[name].state_dict(),
                    'num_classes': self.datamanager.num_train_pids
                },
                ckpt_name,
                is_best=is_best
            )
            latest_name = osp.join(save_dir, 'latest.pth')
            if osp.lexists(latest_name):
                os.remove(latest_name)
            os.symlink(ckpt_name, latest_name)

    def set_model_mode(self, mode='train', names=None):
        assert mode in ['train', 'eval', 'test']
        names = self.get_model_names(names)

        for name in names:
            if mode == 'train':
                self.models[name].train()
            else:
                self.models[name].eval()

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self.optims[name].param_groups[0]['lr']

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self.scheds[name] is not None:
                self.scheds[name].step()

    def plot_loss_lr(self, losses, lrs):
        plt.figure()
        plt.ylabel("loss")
        plt.xlabel("learning rate")
        plt.plot(lrs, losses)
        # plt.xscale('log')
        plt.savefig('/home/prokofiev/deep-person-reid/test_images/plot1.png')

    def plot_loss_change(self, losses, lrs, sma=1, n_skip=20, y_lim=(-0.01,0.01)):
        """
        Plots rate of change of the loss function.
        Parameters:
            sched - learning rate scheduler, an instance of LR_Finder class.
            sma - number of batches for simple moving average to smooth out the curve.
            n_skip - number of batches to skip on the left.
            y_lim - limits for the y axis.
        """
        plt.figure()
        derivatives = [0] * (sma + 1)
        for i in range(1 + sma, len(lrs)):
            derivative = (losses[i] - losses[i - sma]) / sma
            derivatives.append(derivative)

        plt.ylabel("d/loss")
        plt.xlabel("learning rate (log scale)")
        i = np.argmin(derivatives[n_skip:])
        print(lrs[i])
        plt.plot(lrs[n_skip:], derivatives[n_skip:])
        # plt.xscale('log')
        # plt.ylim(y_lim)
        plt.savefig('/home/prokofiev/deep-person-reid/test_images/plot2.png')

    def find_lr(
        self,
        lr_find_mode='automatic',
        max_lr=0.01,
        min_lr=0.001,
        num_iter=10,
        num_epoch = 3,
        pretrained = True,
        save_dir='log',
        print_freq=10,
        fixbase_epoch=0,
        open_layers=None,
        start_eval=0,
        eval_freq=-1,
        test_only=False,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        use_metric_cuhk03=False,
        ranks=(1, 5, 10, 20),
        rerank=False,
        **kwargs):
        r"""A  pipeline for learning rate search.

        Args:
            lr_find_mode (str, optional): mode for learning rate finder, "automatic" or "brute_force".
                Default is "automatic".
            max_lr (float): upper bound for leaning rate
            min_lr (float): lower bound for leaning rate
            num_iter (int, optional): number of iterations for searching space. Default is 10.
            pretrained (bool): whether or not the model is pretrained
            save_dir (str): directory to save model.
            max_epoch (int): maximum epoch.
            start_epoch (int, optional): starting epoch. Default is 0.
            print_freq (int, optional): print_frequency. Default is 10.
            fixbase_epoch (int, optional): number of epochs to train ``open_layers`` (new layers)
                while keeping base layers frozen. Default is 0. ``fixbase_epoch`` is counted
                in ``max_epoch``.
            open_layers (str or list, optional): layers (attribute names) open for training.
            start_eval (int, optional): from which epoch to start evaluation. Default is 0.
            eval_freq (int, optional): evaluation frequency. Default is -1 (meaning evaluation
                is only performed at the end of training).
            test_only (bool, optional): if True, only runs evaluation on test datasets.
                Default is False.
            dist_metric (str, optional): distance metric used to compute distance matrix
                between query and gallery. Default is "euclidean".
            normalize_feature (bool, optional): performs L2 normalization on feature vectors before
                computing feature distance. Default is False.
            visrank (bool, optional): visualizes ranked results. Default is False. It is recommended to
                enable ``visrank`` when ``test_only`` is True. The ranked images will be saved to
                "save_dir/visrank_dataset", e.g. "save_dir/visrank_market1501".
            visrank_topk (int, optional): top-k ranked images to be visualized. Default is 10.
            use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
                Default is False. This should be enabled when using cuhk03 classic split.
            ranks (list, optional): cmc ranks to be computed. Default is [1, 5, 10, 20].
            rerank (bool, optional): uses person re-ranking (by Zhong et al. CVPR'17).
                Default is False. This is only enabled when test_only=True.
        """
        print('=> Start learning rate search')

        self.num_batches = len(self.train_loader)

        names = self.get_model_names(None)
        name = names[0]
        current_optimizer = self.optims[name]
        wd = current_optimizer.param_groups[0]['weight_decay']
        model = self.models[name]
        model_device = next(model.parameters()).device
        optimizer = current_optimizer.__class__(model.parameters(), lr=1e-5, weight_decay=wd)

        if lr_find_mode == 'automatic':
            criterion = torch.nn.CrossEntropyLoss()
            lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
            lr_finder.range_test(self.train_loader, end_lr=1., num_iter=self.num_batches, step_mode='exp')
            ax, optim_lr = lr_finder.plot()
            fig = ax.get_figure()
            fig.savefig('/home/prokofiev/deep-person-reid/test_images/plot3.png')
            lr_finder.reset()

            return clip(optim_lr, pretrained=pretrained, backbone_name=model.module.__class__.__name__)

        assert lr_find_mode == 'brute_force'
        acc_store = dict()
        self.start_epoch = 0
        self.max_epoch = num_epoch
        self.fixbase_epoch = fixbase_epoch
        state_cacher = StateCacher(in_memory=True, cache_dir=None)
        state_cacher.store("model", self.models[name].module.state_dict())
        state_cacher.store("optimizer", self.optims[name].state_dict())
        old_model = self.models[name].module.state_dict()
        old_optim = self.optims[name].state_dict()
        range_lr = np.linspace(min_lr, max_lr, num_iter)
        best_acc = 0.0

        for lr in range_lr:
            for param_group in self.optims[name].param_groups:
                param_group["lr"] = round(lr,6)
            for self.epoch in range(self.start_epoch, self.max_epoch):
                cur_top1 = 0.
                self.train(
                        print_freq=print_freq,
                        fixbase_epoch=fixbase_epoch,
                        open_layers=open_layers,
                        lr_finder=True
                        )

                top1 = self.test(
                        self.epoch,
                        dist_metric=dist_metric,
                        normalize_feature=normalize_feature,
                        visrank=visrank,
                        visrank_topk=visrank_topk,
                        save_dir=save_dir,
                        use_metric_cuhk03=use_metric_cuhk03,
                        ranks=ranks,
                        lr_finder=True
                    )
                top1 = round(top1, 4)
                if (self.max_epoch < 5) and (self.epoch == self.max_epoch - 1):
                    acc_store[lr] = top1

                elif (self.max_epoch >= 5) and (self.epoch == self.max_epoch - 1):
                    acc_store[lr] = max(cur_top1, top1)

                cur_top1 = top1

            self.models[name].module.load_state_dict(state_cacher.retrieve("model"))
            self.optims[name].load_state_dict(state_cacher.retrieve("optimizer"))
            self.models[name].to(model_device)

            # break if the results got worse
            cur_acc = acc_store[lr]
            if round((best_acc - cur_acc), 6) >= 2.:
                break
            best_acc = max(best_acc, cur_acc)

        max_acc = 0
        for lr, acc in sorted(acc_store.items()):
            if acc >= (max_acc-0.0005):
                max_acc = acc
                opt_lr = lr

        return opt_lr

    def run(
        self,
        save_dir='log',
        tb_log_dir='',
        max_epoch=0,
        start_epoch=0,
        print_freq=10,
        fixbase_epoch=0,
        open_layers=None,
        start_eval=0,
        eval_freq=-1,
        test_only=False,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        use_metric_cuhk03=False,
        ranks=(1, 5, 10, 20),
        rerank=False,
        **kwargs
    ):
        r"""A unified pipeline for training and evaluating a model.

        Args:
            save_dir (str): directory to save model.
            max_epoch (int): maximum epoch.
            start_epoch (int, optional): starting epoch. Default is 0.
            print_freq (int, optional): print_frequency. Default is 10.
            fixbase_epoch (int, optional): number of epochs to train ``open_layers`` (new layers)
                while keeping base layers frozen. Default is 0. ``fixbase_epoch`` is counted
                in ``max_epoch``.
            open_layers (str or list, optional): layers (attribute names) open for training.
            start_eval (int, optional): from which epoch to start evaluation. Default is 0.
            eval_freq (int, optional): evaluation frequency. Default is -1 (meaning evaluation
                is only performed at the end of training).
            test_only (bool, optional): if True, only runs evaluation on test datasets.
                Default is False.
            dist_metric (str, optional): distance metric used to compute distance matrix
                between query and gallery. Default is "euclidean".
            normalize_feature (bool, optional): performs L2 normalization on feature vectors before
                computing feature distance. Default is False.
            visrank (bool, optional): visualizes ranked results. Default is False. It is recommended to
                enable ``visrank`` when ``test_only`` is True. The ranked images will be saved to
                "save_dir/visrank_dataset", e.g. "save_dir/visrank_market1501".
            visrank_topk (int, optional): top-k ranked images to be visualized. Default is 10.
            use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
                Default is False. This should be enabled when using cuhk03 classic split.
            ranks (list, optional): cmc ranks to be computed. Default is [1, 5, 10, 20].
            rerank (bool, optional): uses person re-ranking (by Zhong et al. CVPR'17).
                Default is False. This is only enabled when test_only=True.
        """
        if visrank and not test_only:
            raise ValueError('visrank can be set to True only if test_only=True')

        if test_only:
            self.test(
                0,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank
            )
            return

        if self.writer is None:
            log_dir = tb_log_dir if len(tb_log_dir) else save_dir
            self.writer = SummaryWriter(log_dir=log_dir)

        # Save zeroth checkpoint
        if self.save_chkpt:
            self.save_model(-1, save_dir)

        time_start = time.time()
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.fixbase_epoch = fixbase_epoch
        print('=> Start training')

        for self.epoch in range(self.start_epoch, self.max_epoch):

            self.train(
                print_freq=print_freq,
                fixbase_epoch=fixbase_epoch,
                open_layers=open_layers
            )

            if (self.epoch + 1) >= start_eval \
               and eval_freq > 0 \
               and (self.epoch+1) % eval_freq == 0 \
               and (self.epoch + 1) != self.max_epoch:

                self.test(
                    self.epoch,
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=visrank,
                    visrank_topk=visrank_topk,
                    save_dir=save_dir,
                    use_metric_cuhk03=use_metric_cuhk03,
                    ranks=ranks
                )
                if self.save_chkpt:
                    self.save_model(self.epoch, save_dir)

        if self.max_epoch > 0:
            print('=> Final test')
            self.test(
                self.epoch,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks
            )
            if self.save_chkpt:
                self.save_model(self.epoch, save_dir)

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))

        if self.writer is not None:
            self.writer.close()

    def train(self, print_freq=10, fixbase_epoch=0, open_layers=None, lr_finder=False):
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        accuracy = AverageMeter()

        self.set_model_mode('train')

        self.two_stepped_transfer_learning(
            self.epoch, fixbase_epoch, open_layers
        )

        self.num_batches = len(self.train_loader)
        end = time.time()
        # print(self.num_batches, self.train_loader)
        for self.batch_idx, data in enumerate(self.train_loader):
            data_time.update(time.time() - end)

            loss_summary, avg_acc = self.forward_backward(data)
            batch_time.update(time.time() - end)

            losses.update(loss_summary)
            accuracy.update(avg_acc)

            if (self.batch_idx + 1) % print_freq == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (self.max_epoch - (self.epoch + 1)) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch: [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'cls acc {accuracy.val:.3f} ({accuracy.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr:.6f}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        accuracy=accuracy,
                        eta=eta_str,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )

            if self.writer is not None and not lr_finder:
                n_iter = self.epoch * self.num_batches + self.batch_idx
                self.writer.add_scalar('Train/time', batch_time.avg, n_iter)
                self.writer.add_scalar('Train/data', data_time.avg, n_iter)
                self.writer.add_scalar('Aux/lr', self.get_current_lr(), n_iter)
                self.writer.add_scalar('Accuracy/train', accuracy.avg, n_iter)
                for name, meter in losses.meters.items():
                    self.writer.add_scalar('Loss/' + name, meter.avg, n_iter)

            end = time.time()

        self.update_lr()

    def forward_backward(self, data):
        raise NotImplementedError

    def test(
        self,
        epoch,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        use_metric_cuhk03=False,
        ranks=(1, 5, 10, 20),
        rerank=False,
        lr_finder = False
    ):
        r"""Tests model on target datasets.

        .. note::

            This function has been called in ``run()``.

        .. note::

            The test pipeline implemented in this function suits both image- and
            video-reid. In general, a subclass of Engine only needs to re-implement
            ``extract_features()`` and ``parse_data_for_eval()`` (most of the time),
            but not a must. Please refer to the source code for more details.
        """
        self.set_model_mode('eval')
        targets = list(self.test_loader.keys())
        top1 = None
        for dataset_name in targets:
            domain = 'source' if dataset_name in self.datamanager.sources else 'target'
            print('##### Evaluating {} ({}) #####'.format(dataset_name, domain))

            for model_name, model in self.models.items():
                if get_model_attr(model, 'classification'):
                    top1 = self._evaluate_classification(
                        model=model,
                        epoch=epoch,
                        data_loader=self.test_loader[dataset_name]['gallery'],
                        model_name=model_name,
                        dataset_name=dataset_name,
                        ranks=ranks,
                        lr_finder = lr_finder
                    )
                elif get_model_attr(model, 'contrastive'):
                    pass
                elif dataset_name == 'lfw':
                    self._evaluate_pairwise(
                        model=model,
                        epoch=epoch,
                        data_loader=self.test_loader[dataset_name]['pairs'],
                        model_name=model_name
                    )
                else:
                    top1 = self._evaluate_reid(
                        model=model,
                        epoch=epoch,
                        model_name=model_name,
                        dataset_name=dataset_name,
                        query_loader=self.test_loader[dataset_name]['query'],
                        gallery_loader=self.test_loader[dataset_name]['gallery'],
                        dist_metric=dist_metric,
                        normalize_feature=normalize_feature,
                        visrank=visrank,
                        visrank_topk=visrank_topk,
                        save_dir=save_dir,
                        use_metric_cuhk03=use_metric_cuhk03,
                        ranks=ranks,
                        rerank=rerank,
                        lr_finder = lr_finder
                    )
        return top1

    @torch.no_grad()
    def _evaluate_classification(self, model, epoch, data_loader, model_name, dataset_name, ranks, lr_finder=False):
        cmc, mAP, norm_cm = metrics.evaluate_classification(data_loader, model, self.use_gpu, ranks)

        if self.writer is not None and not lr_finder:
            self.writer.add_scalar('Val/{}/{}/mAP'.format(dataset_name, model_name), mAP, epoch + 1)
            for i, r in enumerate(ranks):
                self.writer.add_scalar('Val/{}/{}/Rank-{}'.format(dataset_name, model_name, r), cmc[i], epoch + 1)

        print('** Results ({}) **'.format(model_name))
        print('mAP: {:.2%}'.format(mAP))
        for i, r in enumerate(ranks):
            print('Rank-{:<3}: {:.2%}'.format(r, cmc[i]))
        if norm_cm.shape[0] <= 20:
            metrics.show_confusion_matrix(norm_cm)

        return cmc[0]

    @torch.no_grad()
    def _evaluate_pairwise(self, model, epoch, data_loader, model_name):
        same_acc, diff_acc, overall_acc, auc, avg_optimal_thresh = metrics.evaluate_lfw(
            data_loader, model, verbose=False
        )

        if self.writer is not None:
            self.writer.add_scalar('Val/LFW/{}/same_accuracy'.format(model_name), same_acc, epoch + 1)
            self.writer.add_scalar('Val/LFW/{}/diff_accuracy'.format(model_name), diff_acc, epoch + 1)
            self.writer.add_scalar('Val/LFW/{}/accuracy'.format(model_name), overall_acc, epoch + 1)
            self.writer.add_scalar('Val/LFW/{}/AUC'.format(model_name), auc, epoch + 1)

        print('\n** Results ({}) **'.format(model_name))
        print('Accuracy: {:.2%}'.format(overall_acc))
        print('Accuracy on positive pairs: {:.2%}'.format(same_acc))
        print('Accuracy on negative pairs: {:.2%}'.format(diff_acc))
        print('Average threshold: {:.2}'.format(avg_optimal_thresh))

    @torch.no_grad()
    def _evaluate_reid(
        self,
        model,
        epoch,
        dataset_name='',
        query_loader=None,
        gallery_loader=None,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        use_metric_cuhk03=False,
        ranks=(1, 5, 10, 20),
        rerank=False,
        model_name='',
        lr_finder = False
    ):
        def _feature_extraction(data_loader):
            f_, pids_, camids_ = [], [], []
            for batch_idx, data in enumerate(data_loader):
                imgs, pids, camids = self.parse_data_for_eval(data)
                if self.use_gpu:
                    imgs = imgs.cuda()

                features = model(imgs),
                features = features.data.cpu()

                f_.append(features)
                pids_.extend(pids)
                camids_.extend(camids)

            f_ = torch.cat(f_, 0)
            pids_ = np.asarray(pids_)
            camids_ = np.asarray(camids_)

            return f_, pids_, camids_

        qf, q_pids, q_camids = _feature_extraction(query_loader)
        gf, g_pids, g_camids = _feature_extraction(gallery_loader)

        if normalize_feature:
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
        distmat = distmat.numpy()

        if rerank:
            distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
            distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        cmc, mAP = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            use_metric_cuhk03=use_metric_cuhk03
        )

        if self.writer is not None and not lr_finder:
            self.writer.add_scalar('Val/{}/{}/mAP'.format(dataset_name, model_name), mAP, epoch + 1)
            for r in ranks:
                self.writer.add_scalar('Val/{}/{}/Rank-{}'.format(dataset_name, model_name, r), cmc[r - 1], epoch + 1)

        print('** Results ({}) **'.format(model_name))
        print('mAP: {:.2%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))

        if visrank:
            visualize_ranked_results(
                distmat,
                self.datamanager.fetch_test_loaders(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrank_' + dataset_name),
                topk=visrank_topk
            )

        return cmc[0]

    @staticmethod
    def compute_loss(criterion, outputs, targets, **kwargs):
        if isinstance(outputs, (tuple, list)):
            loss = DeepSupervision(criterion, outputs, targets, **kwargs)
        else:
            loss = criterion(outputs, targets, **kwargs)
        return loss

    @staticmethod
    def parse_data_for_train(data, output_dict=False, enable_masks=False, use_gpu=False):
        imgs = data[0]

        obj_ids = data[1]
        if use_gpu:
            imgs = imgs.cuda()
            obj_ids = obj_ids.cuda()

        if output_dict:
            if len(data) > 3:
                dataset_ids = data[3].cuda() if use_gpu else data[3]

                masks = None
                if enable_masks:
                    masks = data[4].cuda() if use_gpu else data[4]

                attr = [record.cuda() if use_gpu else record for record in data[5:]]
                if len(attr) == 0:
                    attr = None
            else:
                dataset_ids = torch.zeros_like(obj_ids)
                masks = None
                attr = None

            return dict(img=imgs, obj_id=obj_ids, dataset_id=dataset_ids, mask=masks, attr=attr)
        else:
            return imgs, obj_ids

    @staticmethod
    def parse_data_for_eval(data):
        imgs = data[0]
        obj_ids = data[1]
        cam_ids = data[2]

        return imgs, obj_ids, cam_ids

    def two_stepped_transfer_learning(self, epoch, fixbase_epoch, open_layers):
        """Two-stepped transfer learning.

        The idea is to freeze base layers for a certain number of epochs
        and then open all layers for training.

        Reference: https://arxiv.org/abs/1611.05244
        """

        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch + 1, fixbase_epoch))

            for model in self.models.values():
                open_specified_layers(model, open_layers, strict=False)
        else:
            for model in self.models.values():
                open_all_layers(model)

def clip(lr, pretrained, backbone_name):
    if not pretrained:
        if (lr >= 1.) or (lr <= 1e-4):
            print("Fail to find lr automaticaly. Lr finder gave either too high ot too low learning rate"
                  "set lr to standart one.")
            return 0.01
        return lr
    print(lr, pretrained, backbone_name)
    if backbone_name == "EfficientNet":
        if (exponent(lr) == 3) and (lr <= 0.0035):
            clipped_lr = lr
        elif (exponent(lr) == 3) and (lr > 0.0035):
            clipped_lr = round(lr / 2, 6)
        elif (exponent(lr) >= 4) and (exponent(lr) <= 1):
            print("Fail to find lr automaticaly. LR Finder gave either too high ot too low learning rate"
                  "set lr to average one for EfficientNet: {}".format(0.003))
            return 0.003
        else:
            clipped_lr = lr / 19.6

    elif backbone_name == "MobileNetV3":
        if (lr <= 0.1 and lr > 0.02):
            k = -843.9371*(lr**2) + 168.3795*lr - 2.1338
            clipped_lr = lr / k
        elif (lr < 0.01 or lr > 0.1):
            print("Fail to find lr automaticaly. LR Finder gave either too high ot too low learning rate"
                  "set lr to average one for MobileNetV3: {}".format(0.013))
            return 0.013
        else:
            clipped_lr = lr
    else:
        print("Unknown backbone, the results could be wrong. LR found by ")
        return lr

    print("Finished searching learning rate. Choosed {} as the best proposed.".format(clipped_lr))
    return clipped_lr

def exponent(n):
    s = '{:.16f}'.format(n).split('.')[1]
    return len(s) - len(s.lstrip('0')) + 1

class StateCacher(object):
    def __init__(self, in_memory, cache_dir=None):
        self.in_memory = in_memory
        self.cache_dir = cache_dir

        if self.cache_dir is None:
            import tempfile

            self.cache_dir = tempfile.gettempdir()
        else:
            if not os.path.isdir(self.cache_dir):
                raise ValueError("Given `cache_dir` is not a valid directory.")

        self.cached = {}

    def store(self, key, state_dict):
        if self.in_memory:
            self.cached.update({key: copy.deepcopy(state_dict)})
        else:
            fn = os.path.join(self.cache_dir, "state_{}_{}.pt".format(key, id(self)))
            self.cached.update({key: fn})
            torch.save(state_dict, fn)

    def retrieve(self, key):
        if key not in self.cached:
            raise KeyError("Target {} was not cached.".format(key))

        if self.in_memory:
            return self.cached.get(key)
        else:
            fn = self.cached.get(key)
            if not os.path.exists(fn):
                raise RuntimeError(
                    "Failed to load state in {}. File doesn't exist anymore.".format(fn)
                )
            state_dict = torch.load(fn, map_location=lambda storage, location: storage)
            return state_dict

    def __del__(self):
        """Check whether there are unused cached files existing in `cache_dir` before
        this instance being destroyed."""

        if self.in_memory:
            return

        for k in self.cached:
            if os.path.exists(self.cached[k]):
                os.remove(self.cached[k])