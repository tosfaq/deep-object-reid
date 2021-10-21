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

import logging
import os
import tempfile

from addict import Dict as ADDict
from typing import Any, Dict, Tuple, List, Optional, Union

import cv2 as cv
import numpy as np
import PIL

from ote_sdk.entities.annotation import Annotation, AnnotationSceneKind
from ote_sdk.entities.id import ID
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.exportable_code.inference import BaseOpenVINOInferencer
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.model import (
    ModelStatus,
    ModelEntity,
    ModelFormat,
    OptimizationMethod,
    ModelPrecision,
)

from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.annotation import AnnotationSceneEntity
from ote_sdk.usecases.tasks.interfaces.optimization_interface import (
    IOptimizationTask,
    OptimizationType,
)

from sc_sdk.entities.annotation import AnnotationScene
from sc_sdk.entities.datasets import Dataset
from sc_sdk.entities.media_identifier import ImageIdentifier

from compression.api import DataLoader
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights, get_nodes_by_type
from compression.pipeline.initializer import create_pipeline

from torchreid.integration.sc.parameters import OTEClassificationParameters

logger = logging.getLogger(__name__)

def get_output(net, outputs, name):
    try:
        key = net.get_ov_name_for_tensor(name)
        assert key in outputs, f'"{key}" is not a valid output identifier'
    except KeyError as err:
        if name not in outputs:
            raise KeyError(f'Failed to identify output "{name}"') from err
        key = name
    return outputs[key]


class OpenVINOClassificationInferencer(BaseOpenVINOInferencer):
    def __init__(
        self,
        hparams: OTEClassificationParameters,
        labels: List[LabelEntity],
        multilabel: bool,
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
        self.pad_value = 0
        self.multilabel = multilabel

    @staticmethod
    def resize_image(image: np.ndarray, size: Tuple[int], keep_aspect_ratio: bool = False) -> np.ndarray:
        if not keep_aspect_ratio:
            img = PIL.Image.fromarray(image).resize(size, resample=PIL.Image.BILINEAR)
            resized_frame = np.array(img, dtype=np.uint8)
        else:
            h, w = image.shape[:2]
            scale = min(size[1] / h, size[0] / w)
            resized_frame = cv.resize(image, None, fx=scale, fy=scale)
        return resized_frame

    def pre_process(self, image: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        resized_image = self.resize_image(image, (self.w, self.h), self.keep_aspect_ratio_resize)
        resized_image = cv.cvtColor(resized_image, cv.COLOR_RGB2BGR)
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

    def post_process(self, prediction: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> AnnotationSceneEntity:
        raw_output = get_output(self.net, prediction, 'reid_embedding').reshape(-1)
        if self.multilabel:
            raw_output = 1. / (1. + np.exp(-1. * raw_output))
            item_labels = []
            for j in range(raw_output.shape[0]):
                if raw_output[j] > 0.5:
                    label = ScoredLabel(label=self.labels[j], probability=raw_output[j])
                    item_labels.append(label)
            anno = [Annotation(Rectangle.generate_full_box(), labels=item_labels)]
        else:
            i = np.argmax(raw_output)
            raw_output = np.exp(raw_output)
            raw_output /= np.sum(raw_output)
            assigned_label = [ScoredLabel(self.labels[i], probability=raw_output[i])]
            anno = [Annotation(Rectangle.generate_full_box(), labels=assigned_label)]
        media_identifier = ImageIdentifier(image_id=ID())

        return AnnotationScene(
            kind=AnnotationSceneKind.PREDICTION,
            media_identifier=media_identifier,
            annotations=anno)

    def forward(self, image: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.model.infer(image)

class OTEOpenVinoDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, inferencer: BaseOpenVINOInferencer):
        super().__init__(config=None)
        self.dataset = dataset
        self.inferencer = inferencer

    def __getitem__(self, index):
        image = self.dataset[index].numpy
        annotation = self.dataset[index].annotation_scene
        inputs, metadata = self.inferencer.pre_process(image)

        return (index, annotation), inputs, metadata

    def __len__(self):
        return len(self.dataset)

class OpenVINOClassificationTask(IInferenceTask, IEvaluationTask, IOptimizationTask):
    def __init__(self, task_environment: TaskEnvironment):
        self.task_environment = task_environment
        self.hparams = self.task_environment.get_hyper_parameters(OTEClassificationParameters)
        self.model = self.task_environment.model
        self.inferencer = self.load_inferencer()

    def load_inferencer(self) -> OpenVINOClassificationInferencer:
        labels = self.task_environment.label_schema.get_labels(include_empty=False)
        is_multilabel = len(self.task_environment.label_schema.get_groups(False)) > 1
        return OpenVINOClassificationInferencer(self.hparams,
                                                labels,
                                                is_multilabel,
                                                self.model.get_data("openvino.xml"),
                                                self.model.get_data("openvino.bin"))

    def infer(self, dataset: Dataset, inference_parameters: Optional[InferenceParameters] = None) -> Dataset:
        from tqdm import tqdm
        for dataset_item in tqdm(dataset):
            dataset_item.annotation_scene = self.inferencer.predict(dataset_item.numpy)
            dataset_item.append_labels(dataset_item.annotation_scene.annotations[0].get_labels())
        return dataset

    def evaluate(self,
                 output_resultset: ResultSetEntity,
                 evaluation_metric: Optional[str] = None):
        return MetricsHelper.compute_accuracy(output_resultset).get_performance()

    def optimize(self,
                 optimization_type: OptimizationType,
                 dataset: Dataset,
                 output_model: ModelEntity,
                 optimization_parameters: Optional[OptimizationParameters]):

        model_name = self.hparams.algo_backend.model_name.replace(' ', '_')
        if optimization_type is not OptimizationType.POT:
            raise ValueError("POT is the only supported optimization type for OpenVino models")

        data_loader = OTEOpenVinoDataLoader(dataset, self.inferencer)

        with tempfile.TemporaryDirectory() as tempdir:
            xml_path = os.path.join(tempdir, model_name + ".xml")
            bin_path = os.path.join(tempdir, model_name + ".bin")
            with open(xml_path, "wb") as f:
                f.write(self.model.get_data("openvino.xml"))
            with open(bin_path, "wb") as f:
                f.write(self.model.get_data("openvino.bin"))

            model_config = ADDict({
                'model_name': model_name,
                'model': xml_path,
                'weights': bin_path
            })

            model = load_model(model_config)

            if get_nodes_by_type(model, ['FakeQuantize']):
                logger.warning("Model is already optimized by POT")
                output_model.model_status = ModelStatus.FAILED
                return

        engine_config = ADDict({
            'device': 'CPU'
        })

        stat_subset_size = self.hparams.pot_parameters.stat_subset_size
        preset = self.hparams.pot_parameters.preset.name.lower()

        algorithms = [
            {
                'name': 'DefaultQuantization',
                'params': {
                    'target_device': 'ANY',
                    'preset': preset,
                    'stat_subset_size': min(stat_subset_size, len(data_loader))
                }
            }
        ]

        engine = IEEngine(config=engine_config, data_loader=data_loader, metric=None)

        pipeline = create_pipeline(algorithms, engine)

        compressed_model = pipeline.run(model)

        compress_model_weights(compressed_model)

        with tempfile.TemporaryDirectory() as tempdir:
            save_model(compressed_model, tempdir, model_name=model_name)
            with open(os.path.join(tempdir, model_name + ".xml"), "rb") as f:
                output_model.set_data("openvino.xml", f.read())
            with open(os.path.join(tempdir, model_name + ".bin"), "rb") as f:
                output_model.set_data("openvino.bin", f.read())

        # set model attributes for quantized model
        output_model.model_status = ModelStatus.SUCCESS
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = OptimizationType.POT
        output_model.optimization_methods = [OptimizationMethod.QUANTIZATION]
        output_model.precision = [ModelPrecision.INT8]

        self.model = output_model
        self.inferencer = self.load_inferencer()
