import logging
import os
import math
from typing import List, Optional
from copy import deepcopy

import torchreid
from torchreid.ops import DataParallel
from torchreid.apis.training import run_lr_finder, run_training
from torchreid.utils import load_pretrained_weights, set_random_seed
from scripts.default_config import (get_default_config,
                                    imagedata_kwargs,
                                    lr_scheduler_kwargs, model_kwargs,
                                    optimizer_kwargs, merge_from_files_with_base)
from torchreid.integration.sc.monitors import StopCallback, DefaultMetricsMonitor
from torchreid.integration.sc.utils import (OTEClassificationDataset, TrainingProgressCallback,
                                            InferenceProgressCallback, get_actmap, preprocess_features_for_actmap)
from torchreid.integration.sc.inference_task import OTEClassificationInferenceTask

from ote_sdk.entities.metrics import (MetricsGroup, CurveMetric, LineChartInfo,
                                      InfoMetric, VisualizationInfo, VisualizationType,
                                      Performance, ScoreMetric)
from ote_sdk.entities.train_parameters import TrainParameters, default_progress_callback
from ote_sdk.usecases.tasks.interfaces.training_interface import ITrainingTask
from ote_sdk.entities.model import ModelEntity, ModelStatus

from sc_sdk.entities.datasets import Dataset, Subset

logger = logging.getLogger(__name__)


class OTEClassificationTrainingTask(OTEClassificationInferenceTask, ITrainingTask):

    def cancel_training(self):
        """
        Called when the user wants to abort training.
        In this example, this is not implemented.

        :return: None
        """
        logger.info("Cancel training requested.")
        self.stop_callback.stop()

    def _generate_training_metrics_group(self) -> Optional[List[MetricsGroup]]:
        """
        Parses the classification logs to get metrics from the latest training run
        :return output List[MetricsGroup]
        """
        output: List[MetricsGroup] = []

        # Model architecture
        architecture = InfoMetric(name='Model architecture', value=self._model_name)
        visualization_info_architecture = VisualizationInfo(name="Model architecture",
                                                            visualisation_type=VisualizationType.TEXT)
        output.append(MetricsGroup(metrics=[architecture],
                                   visualization_info=visualization_info_architecture))

        # Learning curves
        if self.metrics_monitor is not None:
            for key in self.metrics_monitor.get_metric_keys():
                metric_curve = CurveMetric(xs=self.metrics_monitor.get_metric_timestamps(key),
                                           ys=self.metrics_monitor.get_metric_values(key), name=key)
                visualization_info = LineChartInfo(name=key, x_axis_label="Timestamp", y_axis_label=key)
                output.append(MetricsGroup(metrics=[metric_curve], visualization_info=visualization_info))

        return output

    def train(self, dataset: Dataset, output_model: ModelEntity, train_parameters: Optional[TrainParameters] = None):
        """ Trains a model on a dataset """

        configurable_parameters = self._hyperparams

        if train_parameters is not None and train_parameters.train_on_empty_model:
            train_model = self._create_model(self._cfg)
        else:
            train_model = deepcopy(self._model)

        self._cfg.train.batch_size = configurable_parameters.learning_parameters.batch_size
        self._cfg.test.batch_size = max(1, configurable_parameters.learning_parameters.batch_size // 2)
        self._cfg.train.max_epoch = configurable_parameters.learning_parameters.max_num_epochs

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

        init_acc, final_acc = run_training(self._cfg, datamanager, train_model, optimizer,
                                           scheduler, extra_device_ids, self._cfg.train.lr,
                                           tb_writer=self.metrics_monitor, perf_monitor=time_monitor,
                                           stop_callback=self.stop_callback)

        improved = final_acc > init_acc
        training_metrics = self._generate_training_metrics_group()

        if self._cfg.use_gpu:
            train_model = train_model.module

        self.metrics_monitor.close()
        if self.stop_callback.check_stop():
            logger.info('Training cancelled.')
            return

        if improved or self._task_environment.model is None:
            if improved:
                logger.info("Training finished, and it has an improved model")
            else:
                logger.info("First training round, saving the model.")
            load_pretrained_weights(self._model, os.path.join(self._scratch_space, 'best.pth'))
            self.save_model(output_model)
            output_model.model_status = ModelStatus.SUCCESS
            performance = Performance(score=ScoreMetric(value=final_acc, name="accuracy"),
                                      dashboard_metrics=training_metrics)
            logger.info(f'FINAL MODEL PERFORMANCE {performance}')
            output_model.performance = performance
        else:
            logger.info("Model performance has not improved while training. No new model has been saved.")
