# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import absolute_import, division, print_function
import os
import os.path as osp
import json

import torch

from ..dataset import ImageDataset


class Classification(ImageDataset):
    """Classification dataset.
    """

    def __init__(self, root='', mode='train', **kwargs):

        self.root = osp.abspath(osp.expanduser(root))
        self.data_dir = osp.dirname(self.root)
        self.annot = self.root

        required_files = [
            self.data_dir, self.annot
        ]
        self.check_before_run(required_files)

        if mode == 'train':
            train, classes = self.load_annotation(
                self.annot,
                self.data_dir,
            )
            test = []

        elif mode == 'test':
            test, classes = self.load_annotation(
                self.annot,
                self.data_dir,
            )
            train = []

        else:
            classes = []
            train, test = [], []

        super(Classification, self).__init__(train, test, mode=mode, **kwargs)
        self.classes = classes

    @staticmethod
    def load_annotation(annot_path, data_dir):
        out_data = []
        classes_from_data = set()
        predefined_classes = []
        for line in open(annot_path):
            parts = line.strip().split(' ')
            if parts[0] == "classes":
                predefined_classes = parts[1].split(',')
                continue
            if len(parts) != 2:
                print("line doesn't fits pattern. Expected: 'relative_path/to/image label'")
                continue
            rel_image_path, label_str = parts
            full_image_path = osp.join(data_dir, rel_image_path)
            if not osp.exists(full_image_path):
                print(f"{full_image_path}: doesn't exist. Please check path or file")
                continue

            label = int(label_str)
            classes_from_data.add(label)
            out_data.append((full_image_path, label))
        classes = predefined_classes if predefined_classes else classes_from_data
        class_to_idx = {cls: indx for indx, cls in enumerate(classes)}
        return out_data, class_to_idx


class ExternalDatasetWrapper(ImageDataset):
    def __init__(self, data_provider, mode='train', **kwargs):

        self.data_provider = data_provider

        if mode == 'train':
            train, classes = self.load_annotation(
                self.data_provider
            )
            test = []
        elif mode == 'test':
            test, classes = self.load_annotation(
                self.data_provider
            )
            train = []
        else:
            classes = []
            train, test = [], []

        super().__init__(train, test, mode=mode, **kwargs)


        # restore missing classes in train
        if mode == 'train':
            for i, _ in enumerate(data_provider.get_classes()):
                if i not in self.data_counts:
                    self.data_counts[i] = 0
        self.num_train_ids = len(data_provider.get_classes())
        self.classes = classes

    def __len__(self):
        return len(self.data_provider)

    def get_input(self, idx: int):
        img = self.data_provider[idx]['img']
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, idx: int):
        input_image = self.get_input(idx)
        label = self.data_provider[idx]['label']
        if isinstance(label, (tuple, list)): # when multi-label classification is available
            targets = torch.zeros(self.num_train_ids)
            for obj in label:
                idx = int(obj)
                if idx >= 0:
                    targets[idx] = 1
            label = targets
        else:
            label = int(label)
        return input_image, label

    @staticmethod
    def load_annotation(data_provider):

        all_classes = sorted(data_provider.get_classes())
        class_to_idx = {all_classes[i]: i for i in range(len(all_classes))}

        all_annotation = data_provider.get_annotation()
        out_data = []
        for item in all_annotation:
            out_data.append(('', item['label']))

        return out_data, class_to_idx


class ClassificationImageFolder(ImageDataset):
    """Classification dataset representing raw folders without annotation files.
    """

    def __init__(self, root='', mode='train', filter_classes=None, **kwargs):
        self.root = root
        self.check_before_run(self.root)
        if mode == 'train':
            train, classes = self.load_annotation(
                self.root, filter_classes
            )
            test = []
        elif mode == 'test':
            test, classes = self.load_annotation(
                self.root, filter_classes
            )
            train = []
        else:
            classes = []
            train, test = [], []

        super().__init__(train, test, mode=mode, **kwargs)

        self.classes = classes


    @staticmethod
    def load_annotation(data_dir, filter_classes=None):
        ALLOWED_EXTS = ('.jpg', '.jpeg', '.png', '.gif')
        def is_valid(filename):
            return not filename.startswith('.') and filename.lower().endswith(ALLOWED_EXTS)

        def find_classes(dir, filter_names=None):
            if filter_names:
                classes = [d.name for d in os.scandir(dir) if d.is_dir() and d.name in filter_names]
            else:
                classes = [d.name for d in os.scandir(dir) if d.is_dir()]
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            return class_to_idx

        class_to_idx = find_classes(data_dir, filter_classes)

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
                        out_data.append((path, class_index))

        if not len(out_data):
            print('Failed to locate images in folder ' + data_dir + f' with extensions {ALLOWED_EXTS}')

        return out_data, class_to_idx


class MultiLabelClassification(ImageDataset):
    """Multi label classification dataset.
    """

    def __init__(self, root='', mode='train', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.data_dir = osp.dirname(self.root)
        self.annot = self.root

        required_files = [
            self.data_dir, self.annot
        ]
        self.check_before_run(required_files)
        if mode == 'train':
            train, classes = self.load_annotation(
                self.annot,
                self.data_dir,
            )
            test = []
        elif mode == 'test':
            test, classes = self.load_annotation(
                self.annot,
                self.data_dir,
            )
            train = []
        else:
            classes = []
            train, test = [], []


        super(MultiLabelClassification, self).__init__(train, test, mode=mode, **kwargs)
        self.classes = classes

    @staticmethod
    def load_annotation(annot_path, data_dir):
        out_data = []
        with open(annot_path) as f:
            annotation = json.load(f)
            classes = sorted(annotation['classes'])
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            images_info = annotation['images']
            img_wo_objects = 0
            for img_info in images_info:
                rel_image_path, img_labels = img_info
                full_image_path = osp.join(data_dir, rel_image_path)
                labels_idx = [class_to_idx[lbl] for lbl in img_labels if lbl in class_to_idx]
                assert full_image_path
                if not labels_idx:
                    img_wo_objects += 1
                out_data.append((full_image_path, tuple(labels_idx)))
        if img_wo_objects:
            print(f'WARNING: there are {img_wo_objects} images without labels and will be treated as negatives')
        if create_adj_matrix:
            print('here', thau)
            matrix = prepare_adj_matrix(classes, out_data, thau)
            np.save("voc_adj_matrix", matrix)
        return out_data, class_to_idx

    @staticmethod
    def prepare_word_embedings(label_set, vectors_path):
        with open(vectors_path, 'r') as f:
            vectors = {}
            for line in f:
                vals = line.rstrip().split(' ')
                vectors[vals[0]] = [float(x) for x in vals[1:]]
        word_embedings = []
        for label in label_set:
            if label not in vectors:
                print(f'label: {label} is out of dictionary!\n')
                word_embedings.append(vectors['<unk>'])
            word_embedings.append(vectors[label])

        return word_embedings

    @staticmethod
    def prepare_adj_matrix(label_set, out_data):
        num_classes = len(label_set)
        M = np.zeros((num_classes,num_classes))
        for item in out_data:
            item_labels = item[1]
            if len(item_labels) > 1:
                for pair in permutations(item_labels, 2):
                    i,j = pair
                    M[i,j] += 1

        return M
