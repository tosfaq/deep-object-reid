
import math
import datetime
import os
from os import path as osp
from typing import Callable, Dict, List, Optional, Tuple, Union
from copy import deepcopy

import numpy as np
import cv2 as cv
import torch
import torch.utils as utils
from torch import nn
from torch.utils.data import Dataset as TorchDataset
from PIL import Image

from ote_sdk.entities.shapes.box import Box
from sc_sdk.entities.annotation import Annotation, AnnotationScene, AnnotationSceneKind, NullMediaIdentifier
from sc_sdk.entities.datasets import Dataset, DatasetItem, NullDataset, Subset
from ote_sdk.entities.label import ScoredLabel, LabelEntity, Color
from sc_sdk.entities.label import distinct_colors
from sc_sdk.logging import logger_factory
from sc_sdk.usecases.reporting.callback import Callback
from sc_sdk.entities.dataset_storage import NullDatasetStorage
from sc_sdk.entities.label_schema import (LabelGroup, LabelGroupType,
                                          LabelSchema)

from torchreid.models.common import ModelInterface


def generate_batch_indices(count, batch_size):
    for i in range(math.ceil(count / batch_size)):
        yield slice(i * batch_size, (i + 1) * batch_size)


logger = logger_factory.get_logger("TorchClassificationInstance")


class OTEClassificationDataset():
    def __init__(self, ote_dataset, labels):
        super().__init__()
        self.ote_dataset = ote_dataset
        self.labels = labels
        self.annotation = []

        for i in range(len(self.ote_dataset)):
            if self.ote_dataset[i].annotation.get_labels():
                label = self.ote_dataset[i].annotation.get_labels()[0]
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
                 train_data_root=None,
                 val_data_root=None,
                 test_data_root=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.data_roots = {}
        self.data_roots[Subset.TRAINING] = train_data_root
        self.data_roots[Subset.VALIDATION] = val_data_root
        self.data_roots[Subset.TESTING] = test_data_root
        self.annotations = {}
        for k, v in self.data_roots.items():
            if v:
                self.data_roots[k] = os.path.abspath(v)
                self.annotations[k] = self._load_annotation(self.data_roots[k])

        self.labels = None
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
        for subset in (Subset.TRAINING, Subset.VALIDATION, Subset.TESTING):
            labels = list(self.annotations[subset][1].values())
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
        def create_gt_scored_label(label_name):
            return ScoredLabel(label=self.label_name_to_project_label(label_name))

        img = cv.imread(self.data_info[indx][0])
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        image = Image(name=None, numpy=img, dataset_storage=NullDatasetStorage())
        label = create_gt_scored_label(self.data_info[indx][1])
        shapes = [Box.generate_full_box(labels=[ScoredLabel(label)])]
        annotation_scene = AnnotationScene(kind=AnnotationSceneKind.ANNOTATION,
                                           media_identifier=NullMediaIdentifier(),
                                           annotations=shapes)
        datset_item = DatasetItem(image, annotation_scene)
        return datset_item

    def __len__(self) -> int:
        assert self.data_info is not None
        return len(self.data_info)

    def get_labels(self) -> list:
        return self.labels

    def get_subset(self, subset: Subset) -> Dataset:
        dataset = deepcopy(self)
        if dataset.init_as_subset(subset):
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

    label_schema = LabelSchema()
    exclusive_group = LabelGroup(name="labels", labels=not_empty_labels, group_type=LabelGroupType.EXCLUSIVE)
    empty_group = LabelGroup(name="empty", labels=[emptylabel], group_type=LabelGroupType.EMPTY_LABEL)
    label_schema.add_group(exclusive_group)
    label_schema.add_group(empty_group, exclusive_with=[exclusive_group])
    return label_schema


class ClassificationDataloader(TorchDataset):
    """
    Dataloader that generates logits from DatasetItems.
    """

    def __init__(self, dataset: Union[Dataset, List[DatasetItem]],
                 labels: List[LabelEntity],
                 inference_mode: bool = False,
                 augmentation: Callable = None):
        self.dataset = dataset
        self.labels = labels
        self.inference_mode = inference_mode
        self.augmentation = augmentation

    def __len__(self):
        return len(self.dataset)

    def get_input(self, idx: int):
        """
        Return the centered and scaled input tensor for file with 'idx'
        """
        sample = self.dataset[idx].numpy  # This returns 8-bit numpy array of shape (height, width, RGB)

        if self.augmentation is not None:
            img = Image.fromarray(sample)
            img, _ = self.augmentation((img, ''))
        return img

    def __getitem__(self, idx: int):
        """
        Return the input and the an optional encoded target for training with index 'idx'
        """
        input_image = self.get_input(idx)
        _, h, w = input_image.shape

        if self.inference_mode:
            class_num = np.asarray(0)
        else:
            item = self.dataset[idx]
            if len(item.annotation.get_labels()) == 0:
                raise ValueError(
                    f"No labels in annotation found. Annotation: {item.annotation}")
            label = item.annotation.get_labels()[0]
            class_num = self.labels.index(label)
            class_num = np.asarray(class_num)
        return input_image, class_num.astype(np.float32)

@torch.no_grad()
def predict(dataset_slice: List[DatasetItem], labels: List[LabelEntity], model: ModelInterface,
            transform, device: torch.device) -> List[Tuple[DatasetItem, List[ScoredLabel], np.array]]:
    """
    Predict from a list of 'DatasetItem' using 'model'. Scale image prior to inference to 'image_shape'
    :return: Return a list of tuple instances, that hold the resulting DatasetItem, ScoredLabels
    and the saliency map generated by Gradcam++
    """
    model.eval()
    model.to(device)
    instances_per_image = list()
    logger.info("Predicting {} files".format(len(dataset_slice)))

    d_set = ClassificationDataloader(dataset=dataset_slice, labels=labels, augmentation=transform, inference_mode=True)
    loader = utils.data.DataLoader(d_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    for inputs, y in loader:  # tqdm
        inputs = torch.tensor(inputs, device=device)
        outputs = model(inputs)[0]
        outputs = outputs.cpu().detach().numpy()

        # Multiclass
        for i, output in enumerate(outputs):
            dataset_item = dataset_slice[i]
            class_num = int(np.argmax(output))
            class_prob = float(outputs[i, class_num].squeeze())
            label = ScoredLabel(label=labels[class_num], probability=class_prob)
            scored_labels = [label]
            dataset_item.append_labels(labels=scored_labels)
            instances_per_image.append((dataset_item, scored_labels))

    return instances_per_image

def list_available_models(models_directory):
    available_models = []
    for dirpath, dirnames, filenames in os.walk(models_directory):
        for filename in filenames:
            if filename == 'main_model.yaml':
                available_models.append(dict(
                    name=osp.basename(dirpath),
                    dir=osp.join(models_directory, dirpath)))
    available_models.sort(key=lambda x: x['name'])
    return available_models
