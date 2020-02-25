from __future__ import division, print_function, absolute_import
import os.path as osp
from lxml import etree

from ..dataset import ImageDataset


class AIC20(ImageDataset):
    """AIC20.

    URL: `<https://www.aicitychallenge.org/2020-track2-download>`_

    Dataset statistics:
        - identities: 666.
        - images: 36935 (train) + 1052 (query) + 18290 (gallery).
    """
    dataset_dir = 'aic20_reduced'

    def __init__(self, root='', aic20_simulation_data=False, aic20_simulation_only=False, **kwargs):
        if not aic20_simulation_data and aic20_simulation_only:
            raise ValueError('To use simulation only data it should be switched on')

        self.simulation_data = aic20_simulation_data
        self.simulation_only = aic20_simulation_only
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.data_dir = self.dataset_dir

        self.train_dir = osp.join(self.data_dir, 'image_train')
        self.query_dir = osp.join(self.data_dir, 'image_query')
        self.gallery_dir = osp.join(self.data_dir, 'image_test')
        self.extra_train_dir = osp.join(self.data_dir, 'image_train_simulation')

        self.train_annot = osp.join(self.data_dir, 'train_label.xml')
        self.query_annot = osp.join(self.data_dir, 'query_label.xml')
        self.gallery_annot = osp.join(self.data_dir, 'test_label.xml')
        self.extra_train_annot = osp.join(self.data_dir, 'train_simulation_label.xml')

        required_files = [
            self.data_dir, self.query_dir, self.gallery_dir,
            self.query_annot, self.gallery_annot,
        ]
        if not self.simulation_only:
            required_files += [self.train_dir, self.train_annot]
        if self.simulation_data:
            required_files += [self.extra_train_dir, self.extra_train_annot]
        self.check_before_run(required_files)

        train = self.load_annotation(self.train_annot, self.train_dir)
        query = self.load_annotation(self.query_annot, self.query_dir)
        gallery = self.load_annotation(self.gallery_annot, self.gallery_dir)
        if self.simulation_data:
            extra_train = self.load_annotation(self.extra_train_annot, self.extra_train_dir)
            if self.simulation_only:
                train = extra_train
            else:
                train += extra_train

        train = self.compress_labels(train)

        super(AIC20, self).__init__(train, query, gallery, **kwargs)

    @staticmethod
    def load_annotation(annot_path, data_dir):
        tree = etree.parse(annot_path)
        root = tree.getroot()

        assert len(root) == 1
        items = root[0]

        data = []
        for item in items:
            image_name = item.attrib['imageName']
            full_image_path = osp.join(data_dir, image_name)
            assert osp.exists(full_image_path)

            pid = int(item.attrib['vehicleID'])
            cam_id = int(item.attrib['cameraID'][1:])

            data.append((full_image_path, pid, cam_id))

        return data

    @staticmethod
    def compress_labels(data):
        pid_container = set(pid for _, pid, _ in data)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        out_data = [(image_name, pid2label[pid], cam_id) for image_name, pid, cam_id in data]

        return out_data
