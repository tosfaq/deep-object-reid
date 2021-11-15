import io
import logging
import os
import math
from typing import List, Optional
from copy import deepcopy
import tempfile
import shutil

import torch

import torchreid
from torchreid.ops import DataParallel
from torchreid.apis.export import export_onnx, export_ir
from torchreid.apis.training import run_lr_finder, run_training
from torchreid.utils import load_pretrained_weights, set_random_seed
from scripts.default_config import (get_default_config,
                                    imagedata_kwargs,
                                    lr_scheduler_kwargs, model_kwargs,
                                    optimizer_kwargs, merge_from_files_with_base)
from torchreid.integration.sc.monitors import StopCallback, DefaultMetricsMonitor
from torchreid.integration.sc.utils import (OTEClassificationDataset, TrainingProgressCallback,
                                            InferenceProgressCallback, get_actmap, preprocess_features_for_actmap,
                                            active_score_from_probs, get_multiclass_predictions,
                                            get_multilabel_predictions, sigmoid_numpy, softmax_numpy,
                                            get_empty_label)
from torchreid.integration.sc.parameters import OTEClassificationParameters
from torchreid.metrics.classification import score_extraction

from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.metrics import (LineMetricsGroup, CurveMetric, LineChartInfo,
                                      Performance, ScoreMetric, MetricsGroup)
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters, default_progress_callback
from ote_sdk.usecases.tasks.interfaces.training_interface import ITrainingTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.unload_interface import IUnload
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.configuration import cfg_helper
from ote_sdk.configuration.helper.utils import ids_to_strings
from ote_sdk.entities.model import ModelPrecision
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from ote_sdk.entities.model import ModelEntity, ModelStatus
from ote_sdk.entities.metadata import FloatMetadata, FloatType
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.tensor import TensorEntity
from ote_sdk.entities.result_media import ResultMediaEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.subset import Subset


logger = logging.getLogger(__name__)


class OTEClassificationTask(ITrainingTask, IInferenceTask, IEvaluationTask, IExportTask, IUnload):

    task_environment: TaskEnvironment

    def __init__(self, task_environment: TaskEnvironment):
        logger.info("Loading OTEClassificationTask.")
        self._scratch_space = tempfile.mkdtemp(prefix="ote-cls-scratch-")
        logger.info(f"Scratch space created at {self._scratch_space}")

        self._task_environment = task_environment
        if len(task_environment.get_labels(False)) == 1:
            self._labels = task_environment.get_labels(include_empty=True)
        else:
            self._labels = task_environment.get_labels(include_empty=False)
        self._empty_label = get_empty_label(task_environment)
        self._multilabel = len(task_environment.label_schema.get_groups(False)) > 1

        template_file_path = task_environment.model_template.model_template_path

        base_dir = os.path.abspath(os.path.dirname(template_file_path))

        self._cfg = get_default_config()
        self._patch_config(base_dir)

        if self._multilabel:
            assert self._cfg.model.type == 'multilabel', task_environment.model_template.model_template_path + \
                ' model template does not support multiclass classification'
        else:
            assert self._cfg.model.type == 'classification', task_environment.model_template.model_template_path + \
                ' model template does not support multilabel classification'

        self.device = torch.device("cuda:0") if torch.cuda.device_count() else torch.device("cpu")
        self._model = self._load_model(task_environment.model).to(self.device)

        self.stop_callback = StopCallback()
        self.metrics_monitor = DefaultMetricsMonitor()

    @property
    def _hyperparams(self):
        return self._task_environment.get_hyper_parameters(OTEClassificationParameters)

    def _load_model(self, model: ModelEntity):
        if model is not None:
            # If a model has been trained and saved for the task already, create empty model and load weights here
            buffer = io.BytesIO(model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device('cpu'))

            model = self._create_model(self._cfg, from_scratch=True)

            try:
                load_pretrained_weights(model, pretrained_dict=model_data['model'])
                logger.info("Loaded model weights from Task Environment")
            except BaseException as ex:
                raise ValueError("Could not load the saved model. The model file structure is invalid.") \
                    from ex
        else:
            # If there is no trained model yet, create model with pretrained weights as defined in the model config
            # file.
            model = self._create_model(self._cfg, from_scratch=False)
            logger.info("No trained model in project yet. Created new model with general-purpose pretrained weights.")
        return model

    def _create_model(self, config, from_scratch: bool = False):
        """
        Creates a model, based on the configuration in config
        :param config: deep-object-reid configuration from which the model has to be built
        :param from_scratch: bool, if True does not load any weights
        :return model: Model in training mode
        """
        num_train_classes = len(self._labels)
        model = torchreid.models.build_model(**model_kwargs(config, num_train_classes))
        if self._cfg.model.load_weights:
            load_pretrained_weights(model, self._cfg.model.load_weights)
        return model

    def _patch_config(self, base_dir: str):
        self._cfg = get_default_config()
        if self._multilabel:
            config_file_path = os.path.join(base_dir, 'main_model_multilabel.yaml')
        else:
            config_file_path = os.path.join(base_dir, 'main_model.yaml')
        merge_from_files_with_base(self._cfg, config_file_path)
        self._cfg.use_gpu = torch.cuda.device_count() > 0
        self.num_devices = 1 if self._cfg.use_gpu else 0

        self._cfg.custom_datasets.types = ['external_classification_wrapper', 'external_classification_wrapper']
        self._cfg.custom_datasets.names = ['train', 'val']
        self._cfg.custom_datasets.roots = ['']*2
        self._cfg.data.sources = ['train']
        self._cfg.data.targets = ['val']
        self._cfg.data.save_dir = self._scratch_space

        self._cfg.test.test_before_train = False
        self.num_classes = len(self._labels)

        for i, conf in enumerate(self._cfg.mutual_learning.aux_configs):
            if str(base_dir) not in conf:
                self._cfg.mutual_learning.aux_configs[i] = os.path.join(base_dir, conf)

    def cancel_training(self):
        """
        Called when the user wants to abort training.
        In this example, this is not implemented.

        :return: None
        """
        logger.info("Cancel training requested.")
        self.stop_callback.stop()

    def save_model(self, output_model: ModelEntity):
        buffer = io.BytesIO()
        hyperparams = self._task_environment.get_hyper_parameters(OTEClassificationParameters)
        hyperparams_str = ids_to_strings(cfg_helper.convert(hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        modelinfo = {'model': self._model.state_dict(), 'config': hyperparams_str, 'labels': labels, 'VERSION': 1}
        torch.save(modelinfo, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())

    def infer(self, dataset: DatasetEntity,
              inference_parameters: Optional[InferenceParameters] = None) -> DatasetEntity:
        """
        Perform inference on the given dataset.

        :param dataset: Dataset entity to analyse
        :param inference_parameters: Additional parameters for inference.
            For example, when results are generated for evaluation purposes, Saliency maps can be turned off.
        :return: Dataset that also includes the classification results
        """

        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress
        else:
            update_progress_callback = default_progress_callback

        self._cfg.test.batch_size = max(1, self._hyperparams.learning_parameters.batch_size // 2)
        time_monitor = InferenceProgressCallback(math.ceil(len(dataset) / self._cfg.test.batch_size),
                                                 update_progress_callback)

        labels = [label.name for label in self._labels]
        self._cfg.custom_datasets.roots = [OTEClassificationDataset(dataset, labels, self._multilabel),
                                           OTEClassificationDataset(dataset, labels, self._multilabel)]
        datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs(self._cfg))
        mix_precision_status = self._model.mix_precision
        self._model.mix_precision = False
        self._model.eval()
        self._model.to(self.device)
        targets = list(datamanager.test_loader.keys())
        dump_features = not inference_parameters.is_evaluation
        inference_results, _ = score_extraction(datamanager.test_loader[targets[0]]['query'],
                                                self._model, self._cfg.use_gpu, perf_monitor=time_monitor,
                                                feature_dump_mode='all' if dump_features else 'vecs')
        self._model.mix_precision = mix_precision_status
        if dump_features:
            scores, features, feature_vecs = inference_results
            features = preprocess_features_for_actmap(features)
        else:
            scores, feature_vecs = inference_results

        if self._multilabel:
            scores = sigmoid_numpy(scores)

        for i in range(scores.shape[0]):
            dataset_item = dataset[i]

            if self._multilabel:
                item_labels = get_multilabel_predictions(scores[i], self._labels, activate=False)
                if not item_labels:
                    item_labels = [ScoredLabel(self._empty_label, probability=1.)]
            else:
                scores[i] = softmax_numpy(scores[i])
                item_labels = get_multiclass_predictions(scores[i], self._labels, activate=False)

            dataset_item.append_labels(item_labels)

            active_score = active_score_from_probs(scores[i])
            active_score_media = FloatMetadata(name="Active score", value=active_score,
                                               float_type=FloatType.ACTIVE_SCORE)
            dataset_item.append_metadata_item(active_score_media)
            feature_vec_media = TensorEntity(name="representation_vector", numpy=feature_vecs[i])
            dataset_item.append_metadata_item(feature_vec_media)

            if dump_features:
                actmap = get_actmap(features[i], (dataset_item.width, dataset_item.height))
                saliency_media = ResultMediaEntity(name="Saliency map", type="Saliency_map",
                                                   annotation_scene=dataset_item.annotation_scene, numpy=actmap)
                dataset_item.append_metadata_item(saliency_media)

        return dataset

    def _generate_training_metrics_group(self) -> Optional[List[MetricsGroup]]:
        """
        Parses the classification logs to get metrics from the latest training run
        :return output List[MetricsGroup]
        """
        output: List[MetricsGroup] = []

        # Learning curves
        if self.metrics_monitor is not None:
            for key in self.metrics_monitor.get_metric_keys():
                metric_curve = CurveMetric(xs=self.metrics_monitor.get_metric_timestamps(key),
                                           ys=self.metrics_monitor.get_metric_values(key), name=key)
                visualization_info = LineChartInfo(name=key, x_axis_label="Timestamp", y_axis_label=key)
                output.append(LineMetricsGroup(metrics=[metric_curve], visualization_info=visualization_info))

        return output

    def train(self, dataset: DatasetEntity, output_model: ModelEntity,
              train_parameters: Optional[TrainParameters] = None):
        """ Trains a model on a dataset """

        train_model = deepcopy(self._model)

        self._cfg.train.lr = self._hyperparams.learning_parameters.learning_rate
        self._cfg.train.batch_size = self._hyperparams.learning_parameters.batch_size
        self._cfg.test.batch_size = max(1, self._hyperparams.learning_parameters.batch_size // 2)
        self._cfg.train.max_epoch = self._hyperparams.learning_parameters.max_num_epochs

        if train_parameters is not None:
            update_progress_callback = train_parameters.update_progress
        else:
            update_progress_callback = default_progress_callback
        validation_steps = math.ceil((len(dataset.get_subset(Subset.VALIDATION)) / self._cfg.test.batch_size))
        if self._cfg.train.ema.enable:
            validation_steps *= 2
        time_monitor = TrainingProgressCallback(update_progress_callback, num_epoch=self._cfg.train.max_epoch,
                                                num_train_steps=math.ceil(len(dataset.get_subset(Subset.TRAINING)) /
                                                                          self._cfg.train.batch_size),
                                                num_val_steps=0, num_test_steps=0)

        self.metrics_monitor = DefaultMetricsMonitor()
        self.stop_callback.reset()

        set_random_seed(self._cfg.train.seed)
        train_subset = dataset.get_subset(Subset.TRAINING)
        val_subset = dataset.get_subset(Subset.VALIDATION)
        labels = [label.name for label in self._labels]
        self._cfg.custom_datasets.roots = [OTEClassificationDataset(train_subset, labels, self._multilabel),
                                           OTEClassificationDataset(val_subset, labels, self._multilabel)]
        datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs(self._cfg))

        num_aux_models = len(self._cfg.mutual_learning.aux_configs)

        if self._cfg.use_gpu:
            main_device_ids = list(range(self.num_devices))
            extra_device_ids = [main_device_ids for _ in range(num_aux_models)]
            train_model = DataParallel(train_model, device_ids=main_device_ids,
                                       output_device=0).cuda(main_device_ids[0])
        else:
            extra_device_ids = [None for _ in range(num_aux_models)]

        optimizer = torchreid.optim.build_optimizer(train_model, **optimizer_kwargs(self._cfg))

        if self._cfg.lr_finder.enable:
            scheduler = None
        else:
            scheduler = torchreid.optim.build_lr_scheduler(optimizer, num_iter=datamanager.num_iter,
                                                           **lr_scheduler_kwargs(self._cfg))

        if self._cfg.lr_finder.enable:
            _, train_model, optimizer, scheduler = \
                        run_lr_finder(self._cfg, datamanager, train_model, optimizer, scheduler, None,
                                      rebuild_model=False, gpu_num=self.num_devices, split_models=False)

        _, final_acc = run_training(self._cfg, datamanager, train_model, optimizer,
                                    scheduler, extra_device_ids, self._cfg.train.lr,
                                    tb_writer=self.metrics_monitor, perf_monitor=time_monitor,
                                    stop_callback=self.stop_callback)

        training_metrics = self._generate_training_metrics_group()

        if self._cfg.use_gpu:
            train_model = train_model.module

        self.metrics_monitor.close()
        if self.stop_callback.check_stop():
            logger.info('Training cancelled.')
            return

        logger.info("Training finished.")

        best_snap_path = os.path.join(self._scratch_space, 'best.pth')
        if os.path.isfile(best_snap_path):
            load_pretrained_weights(self._model, best_snap_path)
        self.save_model(output_model)
        output_model.model_status = ModelStatus.SUCCESS
        performance = Performance(score=ScoreMetric(value=final_acc, name="accuracy"),
                                  dashboard_metrics=training_metrics)
        logger.info(f'FINAL MODEL PERFORMANCE {performance}')
        output_model.performance = performance

    def evaluate(
        self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None
    ):
        performance = MetricsHelper.compute_accuracy(output_resultset).get_performance()
        logger.info(f"Computes performance of {performance}")
        output_resultset.performance = performance

    def export(self, export_type: ExportType, output_model: ModelEntity):
        assert export_type == ExportType.OPENVINO
        optimized_model_precision = ModelPrecision.FP32

        with tempfile.TemporaryDirectory() as tempdir:
            optimized_model_dir = os.path.join(tempdir, "dor")
            logger.info(f'Optimized model will be temporarily saved to "{optimized_model_dir}"')
            os.makedirs(optimized_model_dir, exist_ok=True)

            onnx_model_path = os.path.join(optimized_model_dir, 'model.onnx')
            mix_precision_status = self._model.mix_precision
            self._model.mix_precision = False
            export_onnx(self._model.eval(), self._cfg, onnx_model_path)
            self._model.mix_precision = mix_precision_status
            export_ir(onnx_model_path, self._cfg.data.norm_mean, self._cfg.data.norm_std,
                      optimized_model_dir=optimized_model_dir, data_type=optimized_model_precision.name)

            bin_file = [f for f in os.listdir(optimized_model_dir) if f.endswith('.bin')][0]
            xml_file = [f for f in os.listdir(optimized_model_dir) if f.endswith('.xml')][0]
            with open(os.path.join(optimized_model_dir, bin_file), "rb") as f:
                output_model.set_data("openvino.bin", f.read())
            with open(os.path.join(optimized_model_dir, xml_file), "rb") as f:
                output_model.set_data("openvino.xml", f.read())
            output_model.precision = [optimized_model_precision]

    @staticmethod
    def _is_docker():
        """
        Checks whether the task runs in docker container
        :return bool: True if task runs in docker
        """
        path = '/proc/self/cgroup'
        is_in_docker = False
        if os.path.isfile(path):
            with open(path) as f:
                is_in_docker = is_in_docker or any('docker' in line for line in f)
        is_in_docker = is_in_docker or os.path.exists('/.dockerenv')
        return is_in_docker

    def _delete_scratch_space(self):
        """
        Remove model checkpoints and logs
        """
        if os.path.exists(self._scratch_space):
            shutil.rmtree(self._scratch_space, ignore_errors=False)

    def unload(self):
        """
        Unload the task
        """
        self._delete_scratch_space()
        if self._is_docker():
            logger.warning(
                "Got unload request. Unloading models. Throwing Segmentation Fault on purpose")
            import ctypes
            ctypes.string_at(0)
        else:
            logger.warning("Got unload request, but not on Docker. Only clearing CUDA cache")
            torch.cuda.empty_cache()
            logger.warning(f"Done unloading. "
                           f"Torch is still occupying {torch.cuda.memory_allocated()} bytes of GPU memory")
