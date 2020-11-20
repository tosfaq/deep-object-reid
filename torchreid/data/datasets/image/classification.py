from __future__ import division, print_function, absolute_import
import os.path as osp
import os

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


class ClassificationImageFolder(ImageDataset):
    """Classification dataset representing raw folders without annotation files.
    """

    def __init__(self, root='', dataset_id=0, load_masks=False, cl_data_dir='cl', **kwargs):
        if load_masks:
            raise NotImplementedError

        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, cl_data_dir)
        self.data_dir = self.dataset_dir

        self.train_images_dir = osp.join(self.data_dir, 'train')
        self.val_images_dir = osp.join(self.data_dir, 'val')

        required_files = [
            self.data_dir, self.train_images_dir, self.val_images_dir
        ]
        self.check_before_run(required_files)

        train = self.load_annotation(
            self.train_images_dir,
            dataset_id=dataset_id
        )
        gallery = self.load_annotation(
            self.val_images_dir,
            dataset_id=dataset_id
        )
        query = []

        super().__init__(train, query, gallery, **kwargs)


    @staticmethod
    def load_annotation(data_dir, dataset_id=0):
        ALLOWED_EXTS = ('.jpg', '.jpeg', '.png')
        def is_valid(filename):
            return not filename.startswith('.') and filename.lower().endswith(ALLOWED_EXTS)

        def find_classes(dir):
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            return class_to_idx

        class_to_idx = find_classes(data_dir)

        out_data = []
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = osp.join(data_dir, target_class)
            if not osp.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = osp.join(root, fname)
                    if is_valid(path):
                        out_data.append((path, class_index, 0, dataset_id, '', -1, -1))\

        if not len(out_data):
            print('Failed to locate images in folder ' + data_dir + )

        return out_data
