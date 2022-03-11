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
                 num_ids=0,
                 **kwargs):

        self.classes = classes
        self.mixed_cls_heads_info = mixed_cls_heads_info
        self.data = data
        self.transform = transform
        self.verbose = verbose
        self.num_ids = num_ids

    def __getitem__(self, index):
        input_record = self.data[index]

        image = read_image(input_record[0], grayscale=False)
        obj_id = input_record[1]

        if isinstance(obj_id, (tuple, list)): # when multi-label classification is available
            if len(self.mixed_cls_heads_info):
                targets = torch.IntTensor(obj_id)
            else:
                targets = torch.zeros(self.num_ids)
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
