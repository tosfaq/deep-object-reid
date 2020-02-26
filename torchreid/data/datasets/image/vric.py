from __future__ import division, print_function, absolute_import
import os.path as osp

from ..dataset import ImageDataset


class VRIC(ImageDataset):
    """VRIC.

    URL: `<https://qmul-vric.github.io>`_

    Dataset statistics:
        - identities: 5622.
        - images: 54808 (train) + 2811 (query) + 2811 (gallery).
    """
    dataset_dir = 'vric'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.data_dir = self.dataset_dir

        self.train_dir = osp.join(self.data_dir, 'train_images')
        self.query_dir = osp.join(self.data_dir, 'probe_images')
        self.gallery_dir = osp.join(self.data_dir, 'gallery_images')

        self.train_annot = osp.join(self.data_dir, 'vric_train.txt')
        self.query_annot = osp.join(self.data_dir, 'vric_probe.txt')
        self.gallery_annot = osp.join(self.data_dir, 'vric_gallery.txt')

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir,
            self.train_annot, self.query_annot, self.gallery_annot,
        ]
        self.check_before_run(required_files)

        train = self.load_annotation(self.train_annot, self.train_dir)
        query = self.load_annotation(self.query_annot, self.query_dir)
        gallery = self.load_annotation(self.gallery_annot, self.gallery_dir)

        train = self.compress_labels(train)

        super(VRIC, self).__init__(train, query, gallery, **kwargs)

    @staticmethod
    def load_annotation(annot_path, data_dir):
        data = []
        for line in open(annot_path):
            parts = line.replace('\n', '').split(' ')
            assert len(parts) == 3

            image_name, pid_str, cam_id_str = parts

            full_image_path = osp.join(data_dir, image_name)
            assert osp.exists(full_image_path)

            pid = int(pid_str)
            cam_id = int(cam_id_str)

            data.append((full_image_path, pid, cam_id))

        return data

    @staticmethod
    def compress_labels(data):
        pid_container = set(pid for _, pid, _ in data)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        out_data = [(image_name, pid2label[pid], cam_id) for image_name, pid, cam_id in data]

        return out_data
