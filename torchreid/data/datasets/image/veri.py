from __future__ import division, print_function, absolute_import
import os.path as osp
from os import listdir

from ..dataset import ImageDataset


class VeRi(ImageDataset):
    """VeRi-776.

    URL: `<https://github.com/VehicleReId/VeRidataset>`_

    Dataset statistics:
        - identities: 776.
        - images: 37778 (train) + 1678 (query) + 11579 (gallery).
    """
    dataset_dir = 'veri'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.data_dir = self.dataset_dir

        self.train_dir = osp.join(self.data_dir, 'image_train')
        self.train_annot = osp.join(self.data_dir, 'train_label.xml')
        self.query_dir = osp.join(self.data_dir, 'image_query')
        self.gallery_dir = osp.join(self.data_dir, 'image_test')

        required_files = [
            self.data_dir, self.train_annot, self.train_dir, self.query_dir, self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, self.load_annotation(self.train_annot))
        query = self.process_dir(self.query_dir)
        gallery = self.process_dir(self.gallery_dir)

        train = self.compress_labels(train)

        super(VeRi, self).__init__(train, query, gallery, **kwargs)

    @staticmethod
    def load_annotation(annot_file):
        if annot_file is None or not osp.exists(annot_file):
            return None

    @staticmethod
    def process_dir(data_dir, annot=None):
        image_files = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f)) and f.endswith('.jpg')]

        data = []
        for image_file in image_files:
            parts = image_file.replace('.jpg', '').split('_')
            assert len(parts) == 4

            pid_str, cam_id_str, local_num_str, _ = parts

            full_image_path = osp.join(data_dir, image_file)
            pid = int(pid_str)
            cam_id = int(cam_id_str[1:])

            if annot is None:
                data.append((full_image_path, pid, cam_id))
            else:
                record = annot[image_file]
                color_id = record['color_id']
                type_id = record['type_id']

                data.append((full_image_path, pid, cam_id, color_id, type_id))

        return data

    @staticmethod
    def compress_labels(data):
        pid_container = set(pid for _, pid, _ in data)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        out_data = [(image_name, pid2label[pid], cam_id) for image_name, pid, cam_id in data]

        return out_data
