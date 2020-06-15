"""
 Copyright (c) 2018-2020 Intel Corporation
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

import os.path as osp
import os
from torch.utils.data import Dataset
from tqdm import tqdm
import cv2 as cv
import numpy as np

from ..dataset import ImageDataset


class VGGFace2(ImageDataset):
    dataset_dir = 'vggface2'
    dataset_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        required_files = [self.dataset_dir]
        self.check_before_run(required_files)

        self.list_train_path = osp.join(self.dataset_dir, 'all_list.txt')
        train = self.process_dir(self.dataset_dir, self.list_train_path)
        query = []
        gallery = []

        super(VGGFace2, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()

        data = []

        pid_container = set()
        for _, img_info in enumerate(lines):
            pid, _ = osp.split(img_info)
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for img_info in lines:
            img_path = osp.join(dir_path, img_info.strip())
            pid, _ = osp.split(img_info)
            camid = 0
            pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data