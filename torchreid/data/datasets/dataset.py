# Copyright (c) 2018-2021 Kaiyang Zhou
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import absolute_import, division, print_function
import os.path as osp

import torch

from torchreid.utils import read_image


class ImageDataset:
    """A base class representing ImageDataset.

    All other image datasets should subclass it.

    ``__getitem__`` returns an image given index.
    It will return ``img``, ``img_path``
    where ``img`` has shape (channel, height, width). As a result,
    data in each batch has shape (batch_size, channel, height, width).
    """

    def __init__(self,
                 data,
                 transform=None,
                 verbose=True,
                 mixed_cls_heads_info={},
                 classes={},
                 num_classes=None,
                 **kwargs):

        self.classes = {}
        self.data = data
        self.transform = transform
        self.verbose = verbose

        self.num_train_ids = (self.get_num_ids(self.data)
                              if not num_classes
                              else num_classes)
        self.data_counts = self.get_data_counts(self.data)

    def __getitem__(self, index):
        input_record = self.data[index]

        image = read_image(input_record[0], grayscale=False)
        obj_id = input_record[1]

        if isinstance(obj_id, (tuple, list)): # when multi-label classification is available
            if len(self.mixed_cls_heads_info):
                targets = torch.IntTensor(obj_id)
            else:
                targets = torch.zeros(self.num_train_ids)
                for obj in obj_id:
                    targets[obj] = 1
            obj_id = targets

        if self.transform is not None:
            transformed_image = self.transform(image)
        else:
            transformed_image = image

        output_record = (transformed_image, obj_id)

        return output_record

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_data_counts(data):
        counts = {}
        for record in data:
            obj_id = record[1]
            if obj_id not in counts:
                counts[obj_id] = 1
            else:
                counts[obj_id] += 1

        return counts

    @staticmethod
    def get_num_ids(data):
        """Parses data list and returns the number of categories.
        """
        if not data:
            return 0
        ids = set()
        for record in data:
            label = record[1]
            if isinstance(label, (list, tuple)):
                if self.mixed_cls_heads_info:
                    for i in range(self.mixed_cls_heads_info['num_multiclass_heads']):
                        if label[i] >= 0:
                            ids.update([self.mixed_cls_heads_info['head_idx_to_logits_range'][i][0] + label[i]])
                    for i in range(self.mixed_cls_heads_info['num_multilabel_classes']):
                        if label[self.mixed_cls_heads_info['num_multiclass_heads'] + i]:
                            ids.update([self.mixed_cls_heads_info['num_single_label_classes'] + i])
                else:
                    ids.update(set(label))
            else:
                ids.add(label)
        return len(ids)

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
                raise RuntimeError(f'"{fpath}" is not found')
