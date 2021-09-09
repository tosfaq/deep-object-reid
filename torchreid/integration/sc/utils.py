
import datetime
import os
from os import path as osp
from copy import deepcopy
import importlib
import tempfile
import subprocess

import cv2 as cv

from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.label import ScoredLabel, LabelEntity, Color
from ote_sdk.entities.annotation import Annotation, AnnotationSceneKind
from sc_sdk.entities.annotation import AnnotationScene, NullMediaIdentifier
from sc_sdk.entities.datasets import Dataset, DatasetItem, NullDataset, Subset
from sc_sdk.entities.image import Image
from sc_sdk.entities.label import distinct_colors
from sc_sdk.entities.dataset_storage import NullDatasetStorage
from ote_sdk.entities.label_schema import (LabelGroup, LabelGroupType,
                                           LabelSchemaEntity)


class OTEClassificationDataset():
    def __init__(self, ote_dataset, labels):
        super().__init__()
        self.ote_dataset = ote_dataset
        self.labels = labels
        self.annotation = []

        for i in range(len(self.ote_dataset)):
            if self.ote_dataset[i].get_shapes_labels():
                label = self.ote_dataset[i].get_shapes_labels()[0].name
                class_num = self.labels.index(label)
            else:
                class_num = 0
            self.annotation.append({'label': class_num})

    def __getitem__(self, idx):
        sample = self.ote_dataset[idx].numpy  # This returns 8-bit numpy array of shape (height, width, RGB)
        label = self.annotation[idx]['label']
        return {'img': sample, 'label': label}

    def __len__(self):
        return len(self.annotation)

    def get_annotation(self):
        return self.annotation

    def get_classes(self):
        return self.labels


class ClassificationDatasetAdapter(Dataset):
    def __init__(self,
                 train_ann_file=None,
                 train_data_root=None,
                 val_ann_file=None,
                 val_data_root=None,
                 test_ann_file=None,
                 test_data_root=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.data_roots = {}
        if train_data_root:
            self.data_roots[Subset.TRAINING] = train_data_root
        if val_data_root:
            self.data_roots[Subset.VALIDATION] = val_data_root
        if test_data_root:
            self.data_roots[Subset.TESTING] = test_data_root
        self.annotations = {}
        for k, v in self.data_roots.items():
            if v:
                self.data_roots[k] = osp.abspath(v)
                self.annotations[k] = self._load_annotation(self.data_roots[k])

        self.labels = None
        self.label_map = None
        self.set_labels_obtained_from_annotation()
        self.project_labels = None

    @staticmethod
    def _load_annotation(data_dir, filter_classes=None, dataset_id=0):
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
            # class_index = class_to_idx[target_class]
            target_dir = osp.join(data_dir, target_class)
            if not osp.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = osp.join(root, fname)
                    if is_valid(path):
                        out_data.append((path, target_class, 0, dataset_id, '', -1, -1))

        if not len(out_data):
            print('Failed to locate images in folder ' + data_dir + f' with extensions {ALLOWED_EXTS}')

        return out_data, class_to_idx

    def set_labels_obtained_from_annotation(self):
        self.labels = None
        self.label_map = {}
        for subset in self.data_roots:
            self.label_map = self.annotations[subset][1]
            labels = list(self.annotations[subset][1].keys())
            if self.labels and self.labels != labels:
                raise RuntimeError('Labels are different from annotation file to annotation file.')
            self.labels = labels
        assert self.labels is not None

    def set_project_labels(self, project_labels):
        self.project_labels = project_labels

    def label_name_to_project_label(self, label_name):
        return [label for label in self.project_labels if label.name == label_name][0]

    def init_as_subset(self, subset: Subset):
        self.data_info = self.annotations[subset][0]
        return True

    def __getitem__(self, indx) -> DatasetItem:
        if isinstance(indx, slice):
            slice_list = range(self.__len__())[indx]
            return [self._load_item(i) for i in slice_list]

        return self._load_item(indx)

    def _load_item(self, indx: int):
        def create_gt_scored_label(label_name):
            return ScoredLabel(label=self.label_name_to_project_label(label_name), probability=1.0)

        img = cv.imread(self.data_info[indx][0])
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        image = Image(name=None, numpy=img, dataset_storage=NullDatasetStorage())
        label = create_gt_scored_label(self.data_info[indx][1])
        shapes = [Annotation(Rectangle.generate_full_box(), [label])]
        annotation_scene = AnnotationScene(kind=AnnotationSceneKind.ANNOTATION,
                                           media_identifier=NullMediaIdentifier(),
                                           annotations=shapes)
        dataset_item = DatasetItem(image, annotation_scene)
        # dataset_item.append_labels(labels=[label])

        return dataset_item

    def __len__(self) -> int:
        assert self.data_info is not None
        return len(self.data_info)

    def get_labels(self) -> list:
        return self.labels

    def get_subset(self, subset: Subset) -> Dataset:
        dataset = deepcopy(self)
        if dataset.init_as_subset(subset):
            dataset.project_labels = self.project_labels
            return dataset
        return NullDataset()


def generate_label_schema(label_names):
    label_domain = "classification"
    colors = distinct_colors(len(label_names)) if len(label_names) > 0 else []
    not_empty_labels = [LabelEntity(name=name, color=colors[i], domain=label_domain, id=i,
                                    is_empty=False, creation_date=datetime.datetime.now()) for i, name in
                        enumerate(label_names)]
    emptylabel = LabelEntity(name=f"Empty label", color=Color(42, 43, 46),
                       is_empty=True, domain=label_domain, id=len(not_empty_labels),creation_date=datetime.datetime.now())

    label_schema = LabelSchemaEntity()
    exclusive_group = LabelGroup(name="labels", labels=not_empty_labels, group_type=LabelGroupType.EXCLUSIVE)
    empty_group = LabelGroup(name="empty", labels=[emptylabel], group_type=LabelGroupType.EMPTY_LABEL)
    label_schema.add_group(exclusive_group)
    label_schema.add_group(empty_group, exclusive_with=[exclusive_group])
    return label_schema


def get_task_class(path):
    module_name, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def reload_hyper_parameters(model_template):
    """ This function copies template.yaml file and its configuration.yaml dependency to temporal folder.
        Then it re-loads hyper parameters from copied template.yaml file.
        This function should not be used in general case, it is assumed that
        the 'configuration.yaml' should be in the same folder as 'template.yaml' file.
    """

    template_file = model_template.model_template_path
    template_dir = osp.dirname(template_file)
    temp_folder = tempfile.mkdtemp()
    conf_yaml = [dep.source for dep in model_template.dependencies if dep.destination == model_template.hyper_parameters.base_path][0]
    conf_yaml = osp.join(template_dir, conf_yaml)
    subprocess.run(f'cp {conf_yaml} {temp_folder}', check=True, shell=True)
    subprocess.run(f'cp {template_file} {temp_folder}', check=True, shell=True)
    model_template.hyper_parameters.load_parameters(osp.join(temp_folder, 'template.yaml'))
    assert model_template.hyper_parameters.data


def set_values_as_default(parameters):
    for v in parameters.values():
        if isinstance(v, dict) and 'value' not in v:
            set_values_as_default(v)
        elif isinstance(v, dict) and 'value' in v:
            if v['value'] != v['default_value']:
                v['value'] = v['default_value']
