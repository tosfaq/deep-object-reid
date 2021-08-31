# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Any, Dict, Tuple, List, Optional, Union

import cv2 as cv
import numpy as np

from ote_sdk.entities.id import ID
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.label import ScoredLabel
from ote_sdk.entities.shapes.box import Box
from ote_sdk.entities.annotation import Annotation, AnnotationSceneKind
from sc_sdk.entities.annotation import AnnotationScene
from sc_sdk.entities.datasets import Dataset
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from sc_sdk.usecases.exportable_code.inference import BaseOpenVINOInferencer
from sc_sdk.entities.label import Label
from sc_sdk.entities.media_identifier import ImageIdentifier
from sc_sdk.entities.resultset import ResultSet
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask

from torchreid.integration.sc.parameters import OTEClassificationParameters


def get_output(net, outputs, name):
    try:
        key = net.get_ov_name_for_tensor(name)
        assert key in outputs, f'"{key}" is not a valid output identifier'
    except KeyError:
        if name not in outputs:
            raise KeyError(f'Failed to identify output "{name}"')
        key = name
    return outputs[key]


class OpenVINOClassificationInferencer(BaseOpenVINOInferencer):
    def __init__(
        self,
        hparams: OTEClassificationParameters,
        labels: List[Label],
        model_file: Union[str, bytes],
        weight_file: Union[str, bytes, None] = None,
        device: str = "CPU",
        num_requests: int = 1,
    ):
        """
        Inferencer implementation for OTEDetection using OpenVINO backend.
        :param model: Path to model to load, `.xml`, `.bin` or `.onnx` file.
        :param hparams: Hyper parameters that the model should use.
        :param num_requests: Maximum number of requests that the inferencer can make.
            Good value is the number of available cores. Defaults to 1.
        :param device: Device to run inference on, such as CPU, GPU or MYRIAD. Defaults to "CPU".
        """
        super().__init__(model_file, weight_file, device, num_requests)
        self.labels = labels
        self.input_blob_name = 'data'
        self.n, self.c, self.h, self.w = self.net.input_info[self.input_blob_name].tensor_desc.dims
        self.keep_aspect_ratio_resize = False

    @staticmethod
    def resize_image(image: np.ndarray, size: Tuple[int], keep_aspect_ratio: bool = False) -> np.ndarray:
        if not keep_aspect_ratio:
            resized_frame = cv.resize(image, size)
        else:
            h, w = image.shape[:2]
            scale = min(size[1] / h, size[0] / w)
            resized_frame = cv.resize(image, None, fx=scale, fy=scale)
        return resized_frame

    def pre_process(self, image: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        resized_image = self.resize_image(image, (self.w, self.h), self.keep_aspect_ratio_resize)
        meta = {'original_shape': image.shape,
                'resized_shape': resized_image.shape}

        h, w = resized_image.shape[:2]
        if h != self.h or w != self.w:
            resized_image = np.pad(resized_image, ((0, self.h - h), (0, self.w - w), (0, 0)),
                                   mode='constant', constant_values=self.pad_value)
        resized_image = resized_image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        resized_image = resized_image.reshape((self.n, self.c, self.h, self.w))
        dict_inputs = {self.input_blob_name: resized_image}
        return dict_inputs, meta

    def post_process(self, prediction: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> AnnotationScene:
        raw_output = get_output(self.net, prediction, 'reid_embedding').reshape(-1)
        raw_output = np.exp(raw_output)
        raw_output /= np.sum(raw_output)
        i = np.argmax(raw_output)

        assigned_label = [ScoredLabel(self.labels[i], probability=raw_output[i])]
        anno = Annotation(Box.generate_full_box(), labels=assigned_label)
        media_identifier = ImageIdentifier(image_id=ID())

        return AnnotationScene(
            kind=AnnotationSceneKind.PREDICTION,
            media_identifier=media_identifier,
            annotations=anno)

    def forward(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.model.infer(inputs)


class OpenVINODetectionTask(IInferenceTask, IEvaluationTask):
    def __init__(self, task_environment: TaskEnvironment):
        self.task_environment = task_environment
        self.hparams = self.task_environment.get_hyper_parameters(OTEClassificationParameters)
        self.model = self.task_environment.model
        self.inferencer = self.load_inferencer()

    def load_inferencer(self) -> OpenVINOClassificationInferencer:
        labels = self.task_environment.label_schema.get_labels(include_empty=False)
        return OpenVINOClassificationInferencer(self.hparams,
                                                labels,
                                                self.model.get_data("openvino.xml"),
                                                self.model.get_data("openvino.bin"))

    def infer(self, dataset: Dataset, inference_parameters: Optional[InferenceParameters] = None) -> Dataset:
        from tqdm import tqdm
        for dataset_item in tqdm(dataset):
            dataset_item.annotation_scene = self.inferencer.predict(dataset_item.numpy)
        return dataset

    def evaluate(self,
                 output_result_set: ResultSet,
                 evaluation_metric: Optional[str] = None):
        return MetricsHelper.compute_accuracy(output_result_set).get_performance()
