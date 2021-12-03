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

import inspect
import json
import logging
import os
import subprocess
import sys
import tempfile
from shutil import copyfile, copytree
from typing import Any, Dict, List, Optional, Tuple, Union

from addict import Dict as ADDict

import attr

import numpy as np

import ote_sdk.usecases.exportable_code.demo as demo
from ote_sdk.entities.inference_parameters import InferenceParameters, default_progress_callback
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.exportable_code.inference import BaseInferencer
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.usecases.tasks.interfaces.deployment_interface import IDeploymentTask
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.entities.model import (
    ModelStatus,
    ModelEntity,
    ModelFormat,
    OptimizationMethod,
    ModelPrecision,
)

from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.usecases.exportable_code.prediction_to_annotation_converter import ClassificationToAnnotationConverter
from ote_sdk.entities.annotation import AnnotationSceneEntity
from ote_sdk.usecases.tasks.interfaces.optimization_interface import (
    IOptimizationTask,
    OptimizationType,
)
from ote_sdk.entities.datasets import DatasetEntity

from compression.api import DataLoader
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights, get_nodes_by_type
from compression.pipeline.initializer import create_pipeline

from openvino.model_zoo.model_api.models import Model
from openvino.model_zoo.model_api.adapters import create_core, OpenvinoAdapter
from torchreid.integration.sc.parameters import OTEClassificationParameters
from torchreid.integration.sc.utils import get_empty_label

from zipfile import ZipFile
from . import model_wrappers

logger = logging.getLogger(__name__)


class OpenVINOClassificationInferencer(BaseInferencer):
    def __init__(
        self,
        hparams: OTEClassificationParameters,
        labels: List[LabelEntity],
        empty_label: LabelEntity,
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
        self.labels = labels
        self.empty_label = empty_label
        try:
            model_adapter = OpenvinoAdapter(create_core(), model_file, weight_file, device=device, max_num_requests=num_requests)
            label_names = [label.name for label in self.labels]
            self.configuration = {**attr.asdict(hparams.inference_parameters.postprocessing,
                                  filter=lambda attr, value: attr.name not in ['header', 'description', 'type', 'visible_in_ui']),
                                  'multilabel': multilabel, 'empty_label': empty_label, 'labels': label_names}
            self.model = Model.create_model(hparams.inference_parameters.class_name.value, model_adapter, self.configuration)
            self.model.load()
        except ValueError as e:
            print(e)
        self.converter = ClassificationToAnnotationConverter(self.labels)

    def pre_process(self, image: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        return self.model.preprocess(image)

    def post_process(self, prediction: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> AnnotationSceneEntity:
        prediction = self.model.postprocess(prediction, metadata)
        metadata['empty_label'] = self.empty_label
        return self.converter.convert_to_annotation(prediction, metadata)

    def forward(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.model.infer_sync(inputs)


class OTEOpenVinoDataLoader(DataLoader):
    def __init__(self, dataset: DatasetEntity, inferencer: BaseInferencer):
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


class OpenVINOClassificationTask(IDeploymentTask, IInferenceTask, IEvaluationTask, IOptimizationTask):
    def __init__(self, task_environment: TaskEnvironment):
        self.task_environment = task_environment
        self.hparams = self.task_environment.get_hyper_parameters(OTEClassificationParameters)
        self.model = self.task_environment.model
        self.model_name = task_environment.model_template.name.replace(" ", "_").replace('-', '_')
        self.empty_label = get_empty_label(task_environment)
        self.inferencer = self.load_inferencer()

    def load_inferencer(self) -> OpenVINOClassificationInferencer:
        labels = self.task_environment.label_schema.get_labels(include_empty=False)
        is_multilabel = len(self.task_environment.label_schema.get_groups(False)) > 1
        return OpenVINOClassificationInferencer(self.hparams,
                                                labels,
                                                self.empty_label,
                                                is_multilabel,
                                                self.model.get_data("openvino.xml"),
                                                self.model.get_data("openvino.bin"))

    def infer(self, dataset: DatasetEntity,
              inference_parameters: Optional[InferenceParameters] = None) -> DatasetEntity:
        update_progress_callback = default_progress_callback
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress
        dataset_size = len(dataset)
        for i, dataset_item in enumerate(dataset, 1):
            predicted_scene = self.inferencer.predict(dataset_item.numpy)
            dataset_item.append_annotations(predicted_scene.annotations)
            dataset_item.append_labels(dataset_item.annotation_scene.annotations[0].get_labels())
            update_progress_callback(int(i / dataset_size * 100))
        return dataset

    def evaluate(self,
                 output_result_set: ResultSetEntity,
                 evaluation_metric: Optional[str] = None):
        if evaluation_metric is not None:
            logger.warning(f'Requested to use {evaluation_metric} metric,'
                           'but parameter is ignored. Use accuracy instead.')
        output_result_set.performance = MetricsHelper.compute_accuracy(output_result_set).get_performance()

    def deploy(self,
               output_model: ModelEntity) -> None:
        work_dir = os.path.dirname(demo.__file__)
        model_file = inspect.getfile(type(self.inferencer.model))
        parameters = {}
        parameters['name_of_model'] = self.model_name
        parameters['type_of_model'] = self.hparams.inference_parameters.class_name.value
        parameters['converter_type'] = 'CLASSIFICATION'
        parameters['model_parameters'] = self.inferencer.configuration
        name_of_package = parameters['name_of_model'].lower()
        with tempfile.TemporaryDirectory() as tempdir:
            copyfile(os.path.join(work_dir, "setup.py"), os.path.join(tempdir, "setup.py"))
            copyfile(os.path.join(work_dir, "requirements.txt"), os.path.join(tempdir, "requirements.txt"))
            copytree(os.path.join(work_dir, "demo_package"), os.path.join(tempdir, name_of_package))
            config_path = os.path.join(tempdir, name_of_package, "config.json")
            with open(config_path, "w") as f:
                json.dump(parameters, f)
            # generate model.py
            if (inspect.getmodule(self.inferencer.model) in
               [module[1] for module in inspect.getmembers(model_wrappers, inspect.ismodule)]):
                copyfile(model_file, os.path.join(tempdir, name_of_package, "model.py"))
            # create wheel package
            subprocess.run([sys.executable, os.path.join(tempdir, "setup.py"), 'bdist_wheel',
                            '--dist-dir', tempdir, 'clean', '--all'])
            wheel_file_name = [f for f in os.listdir(tempdir) if f.endswith('.whl')][0]

            with ZipFile(os.path.join(tempdir, "openvino.zip"), 'w') as zip:
                zip.writestr(os.path.join("model", "model.xml"), self.model.get_data("openvino.xml"))
                zip.writestr(os.path.join("model", "model.bin"), self.model.get_data("openvino.bin"))
                zip.write(os.path.join(tempdir, "requirements.txt"), os.path.join("python", "requirements.txt"))
                zip.write(os.path.join(tempdir, name_of_package, "sync.py"), os.path.join("python", "demo.py"))
                zip.write(os.path.join(tempdir, wheel_file_name), os.path.join("python", wheel_file_name))
            with open(os.path.join(tempdir, "openvino.zip"), "rb") as file:
                output_model.set_data("demo_package", file.read())

    def optimize(self,
                 optimization_type: OptimizationType,
                 dataset: DatasetEntity,
                 output_model: ModelEntity,
                 optimization_parameters: Optional[OptimizationParameters]):

        if optimization_type is not OptimizationType.POT:
            raise ValueError("POT is the only supported optimization type for OpenVino models")

        data_loader = OTEOpenVinoDataLoader(dataset, self.inferencer)

        with tempfile.TemporaryDirectory() as tempdir:
            xml_path = os.path.join(tempdir, "model.xml")
            bin_path = os.path.join(tempdir, "model.bin")
            with open(xml_path, "wb") as f:
                f.write(self.model.get_data("openvino.xml"))
            with open(bin_path, "wb") as f:
                f.write(self.model.get_data("openvino.bin"))

            model_config = ADDict({
                'model_name': 'openvino_model',
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
            save_model(compressed_model, tempdir, model_name="model")
            with open(os.path.join(tempdir, "model.xml"), "rb") as f:
                output_model.set_data("openvino.xml", f.read())
            with open(os.path.join(tempdir, "model.bin"), "rb") as f:
                output_model.set_data("openvino.bin", f.read())

        # set model attributes for quantized model
        output_model.model_status = ModelStatus.SUCCESS
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = OptimizationType.POT
        output_model.optimization_methods = [OptimizationMethod.QUANTIZATION]
        output_model.precision = [ModelPrecision.INT8]

        self.model = output_model
        self.inferencer = self.load_inferencer()
