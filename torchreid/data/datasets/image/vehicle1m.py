from __future__ import division, print_function, absolute_import
from collections import defaultdict
import os.path as osp

from ..dataset import ImageDataset


class Vehicle1M(ImageDataset):
    """VRIC.

    URL: `<http://www.nlpr.ia.ac.cn/iva/homepage/jqwang/Vehicle1M.htm>`_

    Dataset statistics:
        - identities: 50000.
        - images: 844571 (train).
    """

    dataset_dir = 'vehicle-1m'

    def __init__(self, root='', dataset_id=0, min_num_samples=24, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.data_dir = self.dataset_dir

        self.train_dir = osp.join(self.data_dir, 'image')
        self.train_annot = osp.join(self.data_dir, 'train_list.txt')

        required_files = [
            self.data_dir, self.train_dir, self.train_annot
        ]
        self.check_before_run(required_files)

        train = self.load_annotation(
            self.train_annot,
            self.train_dir,
            dataset_id=dataset_id,
            min_num_samples=min_num_samples
        )
        train = self.compress_labels(train)

        query, gallery = [], []

        super(Vehicle1M, self).__init__(train, query, gallery, **kwargs)

    @staticmethod
    def load_annotation(annot_path, data_dir, dataset_id=0, min_num_samples=1):
        data = defaultdict(list)
        for line in open(annot_path):
            parts = line.strip().split(' ')
            assert len(parts) == 3

            image_file, pid_str, model_id_str = parts

            full_image_path = osp.join(data_dir, image_file)
            if not osp.exists(full_image_path):
                continue

            pid = int(pid_str)
            data[pid].append(full_image_path)

        out_data = []
        for pid, records in data.items():
            if len(records) < min_num_samples:
                continue

            for full_image_path in records:
                out_data.append((full_image_path, pid, 0, dataset_id, '', -1, -1))

        return out_data
