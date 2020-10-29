from __future__ import division, print_function, absolute_import
import os.path as osp

from ..dataset import ImageDataset


class Classification(ImageDataset):
    """Classification dataset.
    """

    def __init__(self, root='', dataset_id=0, load_masks=False, cl_data_dir='cl', cl_version='v1', **kwargs):
        if load_masks:
            raise NotImplementedError

        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, f'{cl_data_dir}_{cl_version}')
        self.data_dir = self.dataset_dir

        self.images_dir = osp.join(self.data_dir, 'images')
        self.train_annot = osp.join(self.data_dir, 'train.txt')
        self.test_annot = osp.join(self.data_dir, 'val.txt')

        required_files = [
            self.data_dir, self.images_dir, self.train_annot, self.test_annot
        ]
        self.check_before_run(required_files)

        train = self.load_annotation(
            self.train_annot,
            self.images_dir,
            dataset_id=dataset_id
        )
        gallery = self.load_annotation(
            self.test_annot,
            self.images_dir,
            dataset_id=dataset_id
        )
        query = []

        super(Classification, self).__init__(train, query, gallery, **kwargs)

    @staticmethod
    def load_annotation(annot_path, data_dir, dataset_id=0):
        out_data = []
        for line in open(annot_path):
            parts = line.strip().split(' ')
            if len(parts) != 2:
                continue

            rel_image_path, label_str = parts

            full_image_path = osp.join(data_dir, rel_image_path)
            if not osp.exists(full_image_path):
                continue

            label = int(label_str)
            out_data.append((full_image_path, label, 0, dataset_id, '', -1, -1))

        return out_data
