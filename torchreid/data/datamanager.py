# Copyright (c) 2018-2021 Kaiyang Zhou
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import absolute_import, division, print_function

import numpy as np
import torch

from torchreid.data.datasets import init_image_dataset
from torchreid.data.sampler import build_train_sampler
from torchreid.data.transforms import build_transforms
from torchreid.utils import worker_init_fn

class DataManager(object):
    r"""Base data manager.

    Args:
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): data mean. Default is None (use imagenet mean).
        norm_std (list or None, optional): data std. Default is None (use imagenet std).
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(
        self,
        sources=None,
        targets=None,
        height=256,
        width=128,
        transforms='random_flip',
        norm_mean=None,
        norm_std=None,
        use_gpu=False,
    ):
        self.sources = sources
        self.targets = targets
        self.height = height
        self.width = width

        self.transform_tr, self.transform_te = build_transforms(
            self.height,
            self.width,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.use_gpu = (torch.cuda.is_available() and use_gpu)

    @property
    def num_train_ids(self):
        """Returns the number of training categories."""
        return self._num_train_ids

    @property
    def data_counts(self):
        """Returns the number of samples for each ID."""
        return self._data_counts

    @staticmethod
    def build_dataset_map(groups):
        out_data = dict()
        for group_id, group in enumerate(groups):
            for name in group:
                out_data[name] = group_id

        return out_data

    @staticmethod
    def to_ordered_list(dict_data):
        keys = sorted(dict_data.keys())
        out_data = [dict_data[key] for key in keys]

        return out_data


class ImageDataManager(DataManager):
    r"""Image data manager.

    Args:
        root (str): root path to datasets.
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): data mean. Default is None (use imagenet mean).
        norm_std (list or None, optional): data std. Default is None (use imagenet std).
        use_gpu (bool, optional): use gpu. Default is True.
        combineall (bool, optional): combine train, test in a dataset for
            training. Default is False.
        batch_size_train (int, optional): number of images in a training batch. Default is 32.
        batch_size_test (int, optional): number of images in a test batch. Default is 32.
        workers (int, optional): number of workers. Default is 4.
        train_sampler (str, optional): sampler. Default is RandomSampler.
    """

    def __init__(
        self,
        root='',
        train_name=None,
        test_name=None,
        height=256,
        width=128,
        transforms='random_flip',
        norm_mean=None,
        norm_std=None,
        use_gpu=True,
        split_id=0,
        combineall=False,
        batch_size_train=32,
        batch_size_test=32,
        correct_batch_size = False,
        workers=4,
        train_sampler='RandomSampler',
        custom_dataset_names=[''],
        custom_dataset_roots=[''],
        custom_dataset_types=[''],
        min_samples_per_id=0,
        num_sampled_packages=1,
        filter_classes=None,
    ):

        super(ImageDataManager, self).__init__(
            sources=train_name,
            targets=test_name,
            height=height,
            width=width,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std,
            use_gpu=use_gpu,
        )

        print('=> Loading train dataset')
        train_dataset = []
        train_dataset = init_image_dataset(
            train_name,
            transform=self.transform_tr,
            mode='train',
            combineall=combineall,
            root=root,
            split_id=split_id,
            custom_dataset_names=custom_dataset_names,
            custom_dataset_roots=custom_dataset_roots,
            custom_dataset_types=custom_dataset_types,
            min_id_samples=min_samples_per_id,
            num_sampled_packages=num_sampled_packages,
            filter_classes=filter_classes,
        )
        train_dataset = sum(train_dataset)

        self._data_counts = self.to_ordered_list(train_dataset.data_counts)
        self._num_train_ids = self.to_ordered_list(train_dataset._num_train_ids)
        assert isinstance(self._num_train_ids, list)
        if correct_batch_size:
            batch_size_train = self.calculate_batch(batch_size_train, len(train_dataset))
        batch_size_train = max(1, min(batch_size_train, len(train_dataset)))
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=build_train_sampler(
                train_dataset.train,
                train_sampler,
            ),
            batch_size=batch_size_train,
            shuffle=False,
            worker_init_fn=worker_init_fn,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=True
        )
        self.num_iter = len(self.train_loader)
        print('=> Loading test (target) dataset')
        self.test_loader = {name: {'test': None} for name in self.targets}
        self.test_dataset = {name: {'test': None} for name in self.targets}

        # build test loader
        test_dataset = init_image_dataset(
            test_name,
            transform=self.transform_te,
            mode='test',
            combineall=combineall,
            root=root,
            split_id=split_id,
            custom_dataset_names=custom_dataset_names,
            custom_dataset_roots=custom_dataset_roots,
            custom_dataset_types=custom_dataset_types,
            filter_classes=filter_classes
        )
        self.test_loader[test_name]['test'] = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=max(min(batch_size_test, len(test_dataset)), 1),
            shuffle=False,
            num_workers=workers,
            worker_init_fn=worker_init_fn,
            pin_memory=self.use_gpu,
            drop_last=False
        )

        print('\n')
        print('  **************** Summary ****************')
        print('  source            : {}'.format(self.sources))
        print('  # source datasets : {}'.format(len(self.sources)))
        print('  # source ids      : {}'.format(sum(self._num_train_ids)))
        print('  # source images   : {}'.format(len(train_dataset)))
        print('  target            : {}'.format(self.targets))
        print('  *****************************************')
        print('\n')

    @staticmethod
    def calculate_batch(cur_batch, data_len):
        ''' This heuristic improves multilabel training on small datasets '''
        if data_len <= 2500:
            return max(int(np.ceil(np.sqrt(data_len) / 2.5)), 6)
        return cur_batch
