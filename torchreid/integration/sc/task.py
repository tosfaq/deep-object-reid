import io
import os
import math
from typing import List, Optional
from copy import deepcopy
import tempfile
import shutil

import torch
import numpy as np

import torchreid
from torchreid.engine import build_engine
from torchreid.optim import LrFinder
from torchreid.ops import DataParallel
from torchreid.apis.export import export_onnx, export_ir
from torchreid.utils import load_pretrained_weights, set_random_seed
from scripts.default_config import (engine_run_kwargs, get_default_config,
                                    imagedata_kwargs, lr_finder_run_kwargs,
                                    lr_scheduler_kwargs, model_kwargs,
                                    optimizer_kwargs, merge_from_files_with_base)
from scripts.script_utils import build_auxiliary_model
from torchreid.integration.sc.monitors import PerformanceMonitor, StopCallback, DefaultMetricsMonitor
from torchreid.integration.sc.utils import OTEClassificationDataset
from torchreid.integration.sc.parameters import OTEClassificationParameters
from torchreid.metrics.classification import score_extraction

from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.label_schema import LabelGroupType
from ote_sdk.entities.metrics import MetricsGroup, CurveMetric, LineChartInfo
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters
from ote_sdk.usecases.tasks.interfaces.training_interface import ITrainingTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.unload_interface import IUnload
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.configuration import cfg_helper
from ote_sdk.configuration.helper.utils import ids_to_strings
from ote_sdk.entities.model import ModelPrecision
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from ote_sdk.entities.label import ScoredLabel
from ote_sdk.entities.model import ModelEntity, ModelStatus

from sc_sdk.entities.resultset import ResultSet
from sc_sdk.entities.datasets import Dataset, Subset
from sc_sdk.logging import logger_factory

logger = logger_factory.get_logger("OTEClassificationTask")


class OTEClassificationTask(ITrainingTask, IInferenceTask, IEvaluationTask, IExportTask, IUnload):

    task_environment: TaskEnvironment

    def __init__(self, task_environment: TaskEnvironment):
        logger.info(f"Loading OTEDetectionTask.")
        self._scratch_space = tempfile.mkdtemp(prefix="ote-cls-scratch-")
        logger.info(f"Scratch space created at {self._scratch_space}")

        self._task_environment = task_environment
        self._hyperparams = hyperparams = task_environment.get_hyper_parameters(OTEClassificationParameters)

        self._model_name = hyperparams.algo_backend.model_name
        self._labels = task_environment.get_labels(False)

        template_file_path = task_environment.model_template.model_template_path

        base_dir = os.path.abspath(os.path.dirname(template_file_path))

        self._cfg = get_default_config()
        self._patch_config(base_dir)

        self.device = torch.device("cuda:0") if torch.cuda.device_count() else torch.device("cpu")
        self._model = self._load_model(task_environment.model).to(self.device)

        # Define monitors
        self.stop_callback = StopCallback()
        self.metrics_monitor = DefaultMetricsMonitor()
        self.perf_monitor = PerformanceMonitor()

    def _load_model(self, model: ModelEntity):
        if model is not None:
            # If a model has been trained and saved for the task already, create empty model and load weights here
            buffer = io.BytesIO(model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device('cpu'))

            model = self._create_model(self._cfg, from_scratch=True)

            try:
                load_pretrained_weights(model, pretrained_dict=model_data)
                logger.info(f"Loaded model weights from Task Environment")
                logger.info(f"Model architecture: {self._model_name}")
            except BaseException as ex:
                raise ValueError("Could not load the saved model. The model file structure is invalid.") \
                    from ex
        else:
            # If there is no trained model yet, create model with pretrained weights as defined in the model config
            # file.
            model = self._create_model(self._cfg, from_scratch=False)
            logger.info(f"No trained model in project yet. Created new model with '{self._model_name}' "
                        f"architecture and general-purpose pretrained weights.")
        return model


    def _create_model(self, сonfig, from_scratch: bool = False):
        """
        Creates a model, based on the configuration in config
        :param config: deep-object-reid configuration from which the model has to be built
        :param from_scratch: bool, if True does not load any weights
        :return model: Model in training mode
        """
        num_train_classes = len(self._labels)
        model = torchreid.models.build_model(**model_kwargs(сonfig, num_train_classes))
        if self._cfg.model.load_weights:
            load_pretrained_weights(model, self._cfg.model.load_weights)
        return model

    def _patch_config(self, base_dir: str):
        self._cfg = get_default_config()
        config_file_path = os.path.join(base_dir, self._hyperparams.algo_backend.model)
        merge_from_files_with_base(self._cfg, config_file_path)
        self._cfg.use_gpu = torch.cuda.device_count() > 0
        self.num_devices = 1 if self._cfg.use_gpu else 0
        self._cfg.model.type = 'classification'
        groups = self._task_environment.label_schema.get_groups()
        label_relation_type = groups[0].group_type
        if label_relation_type == LabelGroupType.EXCLUSIVE and len(groups) == 1:
            pass
        else:
            raise ValueError(f"This task does not support non exclusive label groups or multiple groups")

        self._cfg.custom_datasets.types = ['external_classification_wrapper', 'external_classification_wrapper']
        self._cfg.custom_datasets.names = ['train', 'val']
        self._cfg.custom_datasets.roots = ['']*2
        self._cfg.data.sources = ['train']
        self._cfg.data.targets = ['val']
        self._cfg.data.save_dir
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
        self.stop_callback.stop()

    def get_training_progress(self) -> float:
        """
        Returns the progress for training. Returns -1 if this is not known.

        :return: Float with progress [0.0-100.0] or -1.0 if unknown
        """
        return self.perf_monitor.get_training_progress()

    def save_model(self, output_model: ModelEntity):
        buffer = io.BytesIO()
        hyperparams = self._task_environment.get_hyper_parameters(OTEClassificationParameters)
        hyperparams_str = ids_to_strings(cfg_helper.convert(hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        modelinfo = {'model': self._model.state_dict(), 'config': hyperparams_str, 'labels': labels, 'VERSION': 1}
        torch.save(modelinfo, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())

    def infer(self, dataset: Dataset, inference_parameters: Optional[InferenceParameters] = None) -> Dataset:
        """
        Perform inference on the given dataset.

        :param dataset: Dataset entity to analyse
        :param inference_parameters: Additional parameters for inference.
            For example, when results are generated for evaluation purposes, Saliency maps can be turned off.
        :return: Dataset that also includes the classification results
        """
        labels = [label.name for label in self._labels]
        self._cfg.custom_datasets.roots = [OTEClassificationDataset(dataset, labels),
                                           OTEClassificationDataset(dataset, labels)]
        datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs(self._cfg))
        self._model.eval()
        self._model.to(self.device)
        targets = list(datamanager.test_loader.keys())
        scores, _ = score_extraction(datamanager.test_loader[targets[0]]['query'], self._model, self._cfg.use_gpu)
        labels = np.argmax(scores, axis=1)
        predicted_items = []
        for i in range(labels.shape[0]):
            dataset_item = dataset[i]
            class_idx = labels[i]
            scores[i] = np.exp(scores[i])
            scores[i] /= np.sum(scores[i])
            class_prob = float(scores[i, class_idx].squeeze())
            label = ScoredLabel(label=self._labels[class_idx], probability=class_prob)
            dataset_item.append_labels([label])
            predicted_items.append(dataset_item)

        return Dataset(None, predicted_items)

    def generate_training_metrics_group(self) -> List[MetricsGroup]:
        """
        Create additional metrics for the Dashboard.

        :return: List of MetricGroup
        """
        output = []
        if self.metrics_monitor is not None:
            loss = CurveMetric(ys=self.metrics_monitor.get_metric_values('Loss/loss'), name="Training")
            visualization_info = LineChartInfo(name="Loss curve", x_axis_label="Iteration", y_axis_label="Loss value")
            output.append(MetricsGroup(metrics=[loss], visualization_info=visualization_info))
        return output

    def train(self, dataset: Dataset, output_model: ModelEntity, train_parameters: Optional[TrainParameters] = None):
        """ Trains a model on a dataset """

        configurable_parameters = self._hyperparams

        if train_parameters is not None and train_parameters.train_on_empty_model:
            train_model = self.create_model(self._cfg)
        else:
            train_model = deepcopy(self._model)

        self._cfg.train.batch_size = configurable_parameters.learning_parameters.batch_size
        self._cfg.test.batch_size = max(1, configurable_parameters.learning_parameters.batch_size // 2)
        self._cfg.train.max_epoch = configurable_parameters.learning_parameters.max_num_epochs

        train_steps = math.ceil(len(dataset.get_subset(Subset.TRAINING)) / self._cfg.train.batch_size)
        validation_steps = math.ceil((len(dataset.get_subset(Subset.VALIDATION)) / self._cfg.test.batch_size))
        self.perf_monitor.init(self._cfg.train.max_epoch, train_steps, validation_steps)
        self.metrics_monitor = DefaultMetricsMonitor()
        self.stop_callback.reset()

        set_random_seed(self._cfg.train.seed)
        train_subset = dataset.get_subset(Subset.TRAINING)
        val_subset = dataset.get_subset(Subset.VALIDATION)
        labels = [label.name for label in self._labels]
        self._cfg.custom_datasets.roots = [OTEClassificationDataset(train_subset, labels),
                                           OTEClassificationDataset(val_subset, labels)]
        datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs(self._cfg))

        num_aux_models = len(self._cfg.mutual_learning.aux_configs)

        if self._cfg.use_gpu:
            main_device_ids = list(range(self.num_devices))
            extra_device_ids = [main_device_ids for _ in range(num_aux_models)]
            train_model = DataParallel(train_model, device_ids=main_device_ids, output_device=0).cuda(main_device_ids[0])
        else:
            extra_device_ids = [None for _ in range(num_aux_models)]

        optimizer = torchreid.optim.build_optimizer(train_model, **optimizer_kwargs(self._cfg))

        if self._cfg.lr_finder.enable and self._cfg.lr_finder.mode == 'automatic': # and not parameters.resume_from:
            scheduler = None
        else:
            scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(self._cfg))

        lr = None # placeholder, needed for aux models
        if self._cfg.lr_finder.enable:
            if num_aux_models:
                print("Mutual learning is enabled. Learning rate will be estimated for the main model only.")

            # build new engine
            engine = build_engine(self._cfg, datamanager, train_model, optimizer, scheduler)
            lr_finder = LrFinder(engine=engine, **lr_finder_run_kwargs(self._cfg))
            aux_lr = lr_finder.process()

            print(f"Estimated learning rate: {aux_lr}")
            if self._cfg.lr_finder.stop_after:
                print("Finding learning rate finished. Terminate the training process")
                exit()

            # reload all parts of the training
            # we do not check classification parameters
            # and do not get num_train_classes the second time
            # since it's done above and lr finder cannot change parameters of the datasets
            self._cfg.train.lr = aux_lr
            self._cfg.lr_finder.enable = False
            set_random_seed(self._cfg.train.seed, self._cfg.train.deterministic)

            datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs(self._cfg))
            # train_model = torchreid.models.build_model(**model_kwargs(self._cfg, num_train_classes))
            # train_model, _ = put_main_model_on_the_device(model, cfg.use_gpu, args.gpu_num, num_aux_models, args.split_models)
            optimizer = torchreid.optim.build_optimizer(train_model, **optimizer_kwargs(self._cfg))
            scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(self._cfg))

        if num_aux_models:
            print('Enabled mutual learning between {} models.'.format(num_aux_models + 1))

            models, optimizers, schedulers = [train_model], [optimizer], [scheduler]
            for config_file, device_ids in zip(self._cfg.mutual_learning.aux_configs, extra_device_ids):
                aux_model, aux_optimizer, aux_scheduler = build_auxiliary_model(
                    config_file, self.num_classes, self._cfg.use_gpu, device_ids, lr
                )

                models.append(aux_model)
                optimizers.append(aux_optimizer)
                schedulers.append(aux_scheduler)
        else:
            models, optimizers, schedulers = train_model, optimizer, scheduler

        print('Building {}-engine'.format(self._cfg.loss.name))
        engine = build_engine(self._cfg, datamanager, models, optimizers, schedulers)
        engine.run(**engine_run_kwargs(self._cfg), tb_writer=self.metrics_monitor, perf_monitor=self.perf_monitor,
                   stop_callback=self.stop_callback)

        if self._cfg.use_gpu:
            train_model = train_model.module
        #self.metrics_monitor.close()
        if self.stop_callback.check_stop():
            print('Training has been canceled')

        logger.info("Training finished, and it has an improved model")
        self._model = deepcopy(train_model)
        self.save_model(output_model)
        output_model.model_status = ModelStatus.SUCCESS

        self.progress_monitor = None

    def evaluate(
        self, output_resultset: ResultSet, evaluation_metric: Optional[str] = None
    ):
        performance = MetricsHelper.compute_accuracy(output_resultset).get_performance()
        logger.info(f"Computes performance of {performance}")
        return performance

    def export(self, export_type: ExportType, output_model: ModelEntity):
        assert export_type == ExportType.OPENVINO
        optimized_model_precision = ModelPrecision.FP32

        with tempfile.TemporaryDirectory() as tempdir:
            optimized_model_dir = os.path.join(tempdir, "dor")
            logger.info(f'Optimized model will be temporarily saved to "{optimized_model_dir}"')
            os.makedirs(optimized_model_dir, exist_ok=True)

            onnx_model_path = os.path.join(optimized_model_dir, 'model.onnx')
            export_onnx(self._model.eval(), self._cfg, onnx_model_path)
            export_ir(onnx_model_path, self._cfg.data.norm_mean, self._cfg.data.norm_std,
                      optimized_model_dir, optimized_model_precision.name)

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
