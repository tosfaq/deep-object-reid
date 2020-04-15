from __future__ import division, print_function, absolute_import
import os.path as osp
from os import listdir

from lxml import etree

from ..dataset import ImageDataset


class VeRi(ImageDataset):
    """VeRi-776.

    URL: `<https://github.com/VehicleReId/VeRidataset>`_

    Dataset statistics:
        - identities: 776.
        - images: 37778 (train) + 1678 (query) + 11579 (gallery).
    """
    dataset_dir = 'veri'

    def __init__(self, root='', dataset_id=0, **kwargs):
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

        train = self.build_annotation(self.train_dir,
                                      annot=self.load_annotation(self.train_annot),
                                      dataset_id=dataset_id)
        query = self.build_annotation(self.query_dir,
                                      dataset_id=dataset_id)
        gallery = self.build_annotation(self.gallery_dir,
                                        dataset_id=dataset_id)

        train = self.compress_labels(train)

        super(VeRi, self).__init__(train, query, gallery, **kwargs)

    @staticmethod
    def load_annotation(annot_file):
        if annot_file is None or not osp.exists(annot_file):
            return None

        tree = etree.parse(annot_file)
        root = tree.getroot()

        assert len(root) == 1
        items = root[0]

        out_data = dict()
        for item in items:
            image_name = item.attrib['imageName']

            pid = int(item.attrib['vehicleID'])
            cam_id = int(item.attrib['cameraID'][1:])

            color = int(item.attrib['colorID'])
            object_type = int(item.attrib['typeID'])

            out_data[image_name] = dict(
                pid=pid,
                cam_id=cam_id,
                color_id=color - 1,
                type_id=object_type - 1
            )

        return out_data

    @staticmethod
    def build_annotation(data_dir, annot=None, dataset_id=0):
        image_files = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f)) and f.endswith('.jpg')]

        data = []
        for image_file in image_files:
            parts = image_file.replace('.jpg', '').split('_')
            assert len(parts) == 4

            pid_str, cam_id_str, local_num_str, _ = parts

            full_image_path = osp.join(data_dir, image_file)
            pid = int(pid_str)
            cam_id = int(cam_id_str[1:])
            assert pid >= 0 and cam_id >= 0

            if annot is None:
                data.append((full_image_path, pid, cam_id, dataset_id, -1, -1))
            else:
                if image_file not in annot:
                    color_id, type_id = -1, -1
                else:
                    record = annot[image_file]
                    color_id = record['color_id']
                    type_id = record['type_id']
                    assert color_id >= 0 and type_id >= 0

                data.append((full_image_path, pid, cam_id, dataset_id, color_id, type_id))

        return data

    @staticmethod
    def compress_labels(data):
        pid_container = set(record[1] for record in data)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        out_data = [record[:1] + (pid2label[record[1]],) + record[2:] for record in data]

        return out_data
