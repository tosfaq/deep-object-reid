# Copyright (c) 2018-2021 Kaiyang Zhou
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import absolute_import, division, print_function
import copy
import operator
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict

import numpy as np
import torch

from torchreid.utils import download_url, mkdir_if_missing, read_image


class Dataset:
    """An abstract class representing a Dataset.

    This is the base class for ``ImageDataset``.

    Args:
        train (list): contains tuples of (img_path(s), id).
        test (list): contains tuples of (img_path(s), id).
        transform: transform function.
        mode (str): 'train', 'test'.
        combineall (bool): combines train, test in a dataset for training.
        verbose (bool): show information.
    """
    _junk_ids = []  # contains useless person IDs, e.g. background, false detections

    def __init__(
        self,
        train,
        test,
        transform=None,
        mode='train',
        combineall=False,
        verbose=True,
        min_id_samples=0,
        num_sampled_packages=1,
        **kwargs
    ):
        self.train = train
        self.test = test
        self.transform = transform
        self.mode = mode
        self.combineall = combineall
        self.verbose = verbose

        self.num_sampled_packages = num_sampled_packages
        assert self.num_sampled_packages >= 1

        self.num_train_ids = self.get_num_ids(self.train)
        if len(self.num_train_ids) == 0: # workaround for classification: test is a validation set
            self.num_train_ids = self.get_num_ids(self.test)

        if self.combineall:
            self.combine_all()

        self._cut_train_ids(min_id_samples)
        self.data_counts = self.get_data_counts(self.train)

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'test':
            self.data = self.test
        else:
            raise ValueError('Invalid mode. Got {}, but expected to be '
                             'one of [train | test]'.format(self.mode))

        if self.verbose:
            self.show_summary()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        """Adds two datasets together (only the train set)."""
        updated_train = copy.deepcopy(self.train)

        for record in other.train:
            dataset_id = record[2]

            num_train_ids = 0
            if dataset_id in self.num_train_ids:
                num_train_ids = self.num_train_ids[dataset_id]
            old_obj_id = record[1]
            new_obj_id = old_obj_id + num_train_ids

            updated_record = record[0] + new_obj_id + record[2]
            updated_train.append(updated_record)

        ###################################
        # Things to do beforehand:
        # 1. set verbose=False to avoid unnecessary print
        # 2. set combineall=False because combineall would have been applied
        #    if it was True for a specific dataset, setting it to True will
        #    create new IDs that should have been included
        ###################################

        return ImageDataset(
            updated_train,
            self.test,
            transform=self.transform,
            mode=self.mode,
            combineall=False,
            verbose=False
        )

    def __radd__(self, other):
        """Supports sum([dataset1, dataset2, dataset3])."""
        if other == 0:
            return self
        else:
            return self.__add__(other)

    @staticmethod
    def parse_data(data):
        """Parses data list and returns the number of categories.
        """
        ids = defaultdict(set), defaultdict(set)
        for record in data:
            dataset_id = record[2]
            if isinstance(record[1], (tuple, list)):
                for ids in record[1]:
                    ids[dataset_id].add(ids)
            else:
                ids[dataset_id].add(record[1])
        for dataset_id, dataset_ids in ids.items():
            if len(dataset_ids) != max(dataset_ids) + 1:
                print("WARNING:: There are some categories are missing in this split for this dataset.")
            num_ids = {dataset_id: max(dataset_ids) + 1}
        return num_ids

    def get_num_ids(self, data):
        """Returns the number of training categories."""
        return self.parse_data(data)[0]

    @staticmethod
    def get_data_counts(data):
        counts = dict()
        for record in data:
            dataset_id = record[2]
            if dataset_id not in counts:
                counts[dataset_id] = defaultdict(int)

            obj_id = record[1]
            counts[dataset_id][obj_id] += 1

        return counts

    def get_record(self, index):
        return self.data[index]

    def show_summary(self):
        """Shows dataset statistics."""
        pass

    @staticmethod
    def _get_obj_ids(data, junk_obj_ids, obj_ids=None):
        if obj_ids is None:
            obj_ids = defaultdict(set)

        for record in data:
            obj_id = record[1]
            if obj_id in junk_obj_ids:
                continue

            dataset_id = record[2]
            obj_ids[dataset_id].add(obj_id)

        return obj_ids

    @staticmethod
    def _relabel(data, junk_obj_ids, id2label_map, num_train_ids):
        out_data = []
        for record in data:
            obj_id = record[1]
            if obj_id in junk_obj_ids:
                continue

            dataset_id = record[2]
            ids_shift = num_train_ids[dataset_id] if dataset_id in num_train_ids else 0
            updated_obj_id = id2label_map[dataset_id][obj_id] + ids_shift

            updated_record = record[0] + (updated_obj_id,) + record[2]
            out_data.append(updated_record)

        return out_data

    def _cut_train_ids(self, min_imgs):
        if min_imgs < 2:
            return

        self.train = sorted(self.train, key=operator.itemgetter(1))

        id_counters = {}
        for path, id, _ in self.train:
            if id in id_counters:
                id_counters[id] += 1
            else:
                id_counters[id] = 1
        ids_to_del = set()
        for k in id_counters:
            if id_counters[k] < min_imgs:
                ids_to_del.add(k)

        filtered_train = []
        removed_ids = set()
        for path, id, _ in self.train:
            if id in ids_to_del:
                removed_ids.add(id)
            else:
                filtered_train.append((path, id - len(removed_ids)))

        self.train = filtered_train

        self.num_train_ids = self.get_num_ids(self.train)

    def combine_all(self):
        """Combines train, test in a dataset for training."""
        combined = copy.deepcopy(self.train)

        new_obj_ids = self._get_obj_ids(self.test, self._junk_ids)

        id2label_map = dict()
        for dataset_id, dataset_ids in new_obj_ids.items():
            id2label_map[dataset_id] = {obj_id: i for i, obj_id in enumerate(set(dataset_ids))}

        combined += self._relabel(self.test, self._junk_ids, id2label_map, self.num_train_ids)

        self.train = combined
        self.num_train_ids = self.get_num_ids(self.train)

    def download_dataset(self, dataset_dir, dataset_url):
        """Downloads and extracts dataset.

        Args:
            dataset_dir (str): dataset directory.
            dataset_url (str): url to download dataset.
        """
        if osp.exists(dataset_dir):
            return

        if dataset_url is None:
            raise RuntimeError(
                '{} dataset needs to be manually '
                'prepared, please follow the '
                'document to prepare this dataset'.format(
                    self.__class__.__name__
                )
            )

        print('Creating directory "{}"'.format(dataset_dir))
        mkdir_if_missing(dataset_dir)
        fpath = osp.join(dataset_dir, osp.basename(dataset_url))

        print(
            'Downloading {} dataset to "{}"'.format(
                self.__class__.__name__, dataset_dir
            )
        )
        download_url(dataset_url, fpath)

        print('Extracting "{}"'.format(fpath))
        try:
            tar = tarfile.open(fpath)
            tar.extractall(path=dataset_dir)
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(dataset_dir)
            zip_ref.close()

        print('{} dataset is ready'.format(self.__class__.__name__))

    @staticmethod
    def check_before_run(required_files):
        """Checks if required files exist before going deeper.

        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def __repr__(self):
        num_train_ids = self.parse_data(self.train)
        num_test_ids = self.parse_data(self.test)

        msg = '  ------------------------------\n' \
              '  subset   | # ids | # items\n' \
              '  ------------------------------\n' \
              '  train    | {:5d} | {:7d}\n' \
              '  test    | {:5d} | {:7d}\n' \
              '  -------------------------------\n' \
              '  items: images for the dataset\n'.format(
                  sum(num_train_ids.values()), len(self.train),
                  sum(num_test_ids.values()), len(self.test)
              )

        return msg


class ImageDataset(Dataset):
    """A base class representing ImageDataset.

    All other image datasets should subclass it.

    ``__getitem__`` returns an image given index.
    It will return ``img``, ``img_path``
    where ``img`` has shape (channel, height, width). As a result,
    data in each batch has shape (batch_size, channel, height, width).
    """

    def __init__(self, train, test, **kwargs):
        super(ImageDataset, self).__init__(train, test, **kwargs)
        self.classes = {}

    def __getitem__(self, index):
        input_record = self.data[index]

        image = read_image(input_record[0], grayscale=False)
        obj_id = input_record[1]

        if len(input_record) > 3:
            dataset_id = input_record[2]
            if isinstance(obj_id, (tuple, list)): # when multi-label classification is available
                targets = torch.zeros(self.num_train_ids[dataset_id])
                for obj in obj_id:
                    targets[obj] = 1
                obj_id = targets

            if self.mode == 'train' and self.num_sampled_packages > 1:
                assert self.transform is not None

                transformed_image = []
                for _ in range(self.num_sampled_packages):
                    gen_image = self.transform(image)

                    transformed_image.append(gen_image)

                transformed_image = torch.stack(transformed_image)
            elif self.transform is not None:
                transformed_image = self.transform(image)
            else:
                transformed_image = image

            output_record = (transformed_image, obj_id, dataset_id)
        else:
            if self.mode == 'train' and self.num_sampled_packages > 1:
                assert self.transform is not None

                transformed_image = []
                for _ in range(self.num_sampled_packages):
                    gen_image, _ = self.transform((image, ''))

                    transformed_image.append(gen_image)

                transformed_image = torch.stack(transformed_image)
            elif self.transform is not None:
                transformed_image, _ = self.transform((image, ''))
            else:
                transformed_image = image

            output_record = transformed_image, obj_id

        return output_record

    def show_summary(self):
        num_train_ids = self.parse_data(self.train)
        num_test_ids = self.parse_data(self.test)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  ----------------------------------------')
        print('  subset   | # ids | # images ')
        print('  ----------------------------------------')
        print('  train    | {:5d} | {:8d} | {:9d}'.format(
            sum(num_train_ids.values()), len(self.train)))
        print('  test    | {:5d} | {:8d} | {:9d}'.format(
            sum(num_test_ids.values()), len(self.test)))
        print('  ----------------------------------------')
