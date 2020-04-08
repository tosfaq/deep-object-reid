from __future__ import division, print_function, absolute_import
import os.path as osp
from lxml import etree

import numpy as np

from ..dataset import ImageDataset


class AIC20(ImageDataset):
    """AIC20.

    URL: `<https://www.aicitychallenge.org/2020-track2-download>`_

    Dataset statistics:
        - identities: 666.
        - images: 36935 (train) + 1052 (query) + 18290 (gallery).
    """
    dataset_dir = 'aic20'
    angle_bins = np.array([0, 30, 60, 90, 120, 210, 240, 270, 300, 330, 360], dtype=np.float32)

    def __init__(self, root='', aic20_simulation_data=False, aic20_simulation_only=False,
                 aic20_split=False, **kwargs):
        if not aic20_simulation_data and aic20_simulation_only:
            raise ValueError('To use simulation only data it should be switched on')

        self.simulation_data = aic20_simulation_data
        self.simulation_only = aic20_simulation_only
        self.split_data = aic20_split and self.simulation_data and not self.simulation_only
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

        train = self.load_annotation(self.train_annot, self.train_dir, train_mode=True)
        query = self.load_annotation(self.query_annot, self.query_dir)
        gallery = self.load_annotation(self.gallery_annot, self.gallery_dir)
        if self.simulation_data:
            extra_train = self.load_annotation(self.extra_train_annot, self.extra_train_dir, train_mode=True)

            if self.split_data:
                train = self.compress_labels(train)
                extra_train = self.compress_labels(extra_train)

            if self.simulation_only:
                train = extra_train
            else:
                train += extra_train

        if not self.split_data:
            train = self.compress_labels(train)

        super(AIC20, self).__init__(train, query, gallery, **kwargs)

    def get_num_pids(self, data):
        if self.split_data:
            real_pids = set(record[1] for record in data if record[3] < 0)
            synthetic_pids = set(record[1] for record in data if record[3] >= 0)
            return len(real_pids), len(synthetic_pids)
        else:
            return len(set(record[1] for record in data))

    def load_annotation(self, annot_path, data_dir, train_mode=False):
        tree = etree.parse(annot_path)
        root = tree.getroot()

        assert len(root) == 1
        items = root[0]

        data = list() if train_mode else dict()
        for item in items:
            image_name = item.attrib['imageName']
            full_image_path = osp.join(data_dir, image_name)
            assert osp.exists(full_image_path)

            pid = int(item.attrib['vehicleID'])
            cam_id = int(item.attrib['cameraID'][1:])

            if train_mode:
                color = int(item.attrib['colorID']) if 'colorID' in item.attrib else -1
                object_type = int(item.attrib['typeID']) if 'typeID' in item.attrib else -1
                orientation = float(item.attrib['orientation']) if 'orientation' in item.attrib else None

                quantized_angle = self.quantize_angle(orientation, AIC20.angle_bins)

                record = full_image_path, pid, cam_id, color, object_type, quantized_angle
                data.append(record)
            else:
                image_id = int(image_name.split('.')[0])
                if image_id in data:
                    assert ValueError('Image ID {} is duplicated'.format(image_id))

                record = full_image_path, pid, cam_id
                data[image_id] = record

        if train_mode:
            out_data = data
        else:
            ordered_image_ids = sorted(data.keys())
            out_data = [data[key] for key in ordered_image_ids]

        return out_data

    @staticmethod
    def compress_labels(data):
        pid_container = set(record[1] for record in data)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        out_data = []
        for record in data:
            pid = pid2label[record[1]]
            updated_record = tuple([record[0], pid] + list(record[2:]))
            out_data.append(updated_record)

        return out_data

    @staticmethod
    def quantize_angle(angle, centers):
        if angle is None:
            return -1

        dist = np.abs(centers - float(angle))

        class_id = int(np.argmin(dist))
        if class_id == len(centers) - 1:
            class_id = 0

        return class_id

