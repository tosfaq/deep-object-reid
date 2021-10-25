import logging
import os
import io
import json
import math
import collections
from typing import Optional
import torch
import torchreid

from yacs.config import CfgNode

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
from torchreid.integration.sc.parameters import OTEClassificationParameters
from torchreid.integration.nncf.config import compose_nncf_config
from torchreid.integration.nncf.compression import is_state_nncf
from torchreid.integration.nncf.compression_script_utils import (get_nncf_changes_in_aux_training_config,
                                                                 make_nncf_changes_in_training,
                                                                 make_nncf_changes_in_main_training_config)
from ote_sdk.configuration import cfg_helper
from ote_sdk.configuration.helper.utils import ids_to_strings
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.train_parameters import default_progress_callback
from ote_sdk.usecases.tasks.interfaces.optimization_interface import IOptimizationTask, OptimizationType
from ote_sdk.entities.model import (
    ModelEntity,
    OptimizationMethod,
    ModelPrecision,
)

from sc_sdk.entities.datasets import Dataset, Subset


logger = logging.getLogger(__name__)



def deep_update(source, overrides):
    for key, value in overrides.items():
        if isinstance(value, collections.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            returned = CfgNode(returned)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source



class OTEClassificationNNCFTask(OTEClassificationInferenceTask, IOptimizationTask):

    def __init__(self, task_environment: TaskEnvironment):
        self._compression_ctrl = None
        self._nncf_cfg_path = None
        self._nncf_preset = "nncf_quantization"
        super().__init__(task_environment)

    def _set_attributes_by_hyperparams(self):
        quantization = self._hyperparams.nncf_optimization.enable_quantization
        pruning = self._hyperparams.nncf_optimization.enable_pruning
        if quantization and pruning:
            self._nncf_preset = "nncf_quantization_pruning"
            self._optimization_methods = [OptimizationMethod.QUANTIZATION, OptimizationMethod.FILTER_PRUNING]
            self._precision = [ModelPrecision.INT8]
            return
        if quantization and not pruning:
            self._nncf_preset = "nncf_quantization"
            self._optimization_methods = [OptimizationMethod.QUANTIZATION]
            self._precision = [ModelPrecision.INT8]
            return
        if not quantization and pruning:
            self._nncf_preset = "nncf_pruning"
            self._optimization_methods = [OptimizationMethod.FILTER_PRUNING]
            self._precision = [ModelPrecision.FP32]
            return
        raise RuntimeError('Not selected optimization algorithm')

    def _load_model(self, model: ModelEntity):
        # NNCF parts
        nncf_config_path = os.path.join(self._base_dir, "compression_config.json")

        with open(nncf_config_path) as nncf_config_file:
            common_nncf_config = json.load(nncf_config_file)

        self._set_attributes_by_hyperparams()

        optimization_config = compose_nncf_config(common_nncf_config, [self._nncf_preset])

        max_acc_drop = self._hyperparams.nncf_optimization.maximal_accuracy_degradation
        if "accuracy_aware_training" in optimization_config["nncf_config"]:
            # Update maximal_absolute_accuracy_degradation
            (optimization_config["nncf_config"]["accuracy_aware_training"]
            ["params"]["maximal_absolute_accuracy_degradation"]) = max_acc_drop
        else:
            logger.info("NNCF config has no accuracy_aware_training parameters")

        self._nncf_cfg_path = os.path.join(self._scratch_space, 'nncf_config.json')
        with open(self._nncf_cfg_path, 'w') as outfile:
            json.dump(optimization_config["nncf_config"], outfile)

        #TODO: rewrite deep_update
        deep_update(self._cfg, optimization_config)

        self._cfg.nncf.nncf_config_path = self._nncf_cfg_path
        self._nncf_changes_in_aux_train_config = get_nncf_changes_in_aux_training_config(self._cfg)

        if self._cfg.train.ema.enable:
            #TODO: Fix deepcopy NNCF model
            self._cfg.train.ema.enable = False
            self.warning("Disable use_ema_decay. EMA not supported: failed on self.module = deepcopy(model)")

        if model is not None:
            # If a model has been trained and saved for the task already, create empty model and load weights here
            buffer = io.BytesIO(model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device('cpu'))
            model = self._create_model(self._cfg, from_scratch=True)


            init_checkpoint_path = os.path.join(self._scratch_space, 'init_checkpoint.pth')
            with open(init_checkpoint_path, "wb") as outfile:
                # Copy the BytesIO stream to the output file
                outfile.write(buffer.getbuffer())
            self._cfg.model.load_weights = init_checkpoint_path

            if is_state_nncf(model_data):
                # To use nncf checkpoint, state_dict is not supported yet

                self._compression_ctrl, model, self._cfg, aux_lr, nncf_metainfo = \
                    make_nncf_changes_in_training(model, self._cfg,
                                                  model_data['labels'],
                                                  '')
                logger.info("Loaded model weights from Task Environment and wrapped by NNCF")
            else:
                try:
                    model.load_state_dict(model_data['model'])
                    logger.info(f"Loaded model weights from Task Environment")
                    logger.info(f"Model architecture: {self._model_name}")
                except BaseException as ex:
                    raise ValueError("Could not load the saved model. The model file structure is invalid.") \
                        from ex

        else:
            raise ValueError(f"No trained model in project. NNCF require pretrained weights to compress the model")

        return model

    def optimize(
        self,
        optimization_type: OptimizationType,
        dataset: Dataset,
        output_model: ModelEntity,
        optimization_parameters: Optional[OptimizationParameters] = None,
    ):
        """ Optimize a model on a dataset """
        if optimization_type is not OptimizationType.NNCF:
            raise RuntimeError("NNCF is the only supported optimization")

        if self._compression_ctrl:
            raise RuntimeError("Compress already compressed model is not supported")

        configurable_parameters = self._hyperparams

        train_model = self._model

        # self._cfg.train.batch_size = configurable_parameters.learning_parameters.batch_size
        # self._cfg.test.batch_size = max(1, configurable_parameters.learning_parameters.batch_size // 2)
        # self._cfg.train.max_epoch = configurable_parameters.learning_parameters.max_num_epochs

        if optimization_parameters is not None:
            update_progress_callback = optimization_parameters.update_progress
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


        self._compression_ctrl, train_model, self._cfg, aux_lr, nncf_metainfo = \
            make_nncf_changes_in_training(train_model, self._cfg,
                                          labels, '')

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
                                           should_freeze_aux_models=True,
                                           tb_writer=self.metrics_monitor, perf_monitor=time_monitor,
                                           stop_callback=self.stop_callback)

        self.save_model(output_model)

    def save_model(self, output_model: ModelEntity):
        buffer = io.BytesIO()
        hyperparams = self._task_environment.get_hyper_parameters(OTEClassificationParameters)
        hyperparams_str = ids_to_strings(cfg_helper.convert(hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        modelinfo = {
            'model': self._model.state_dict(),
            'compression_state': self._compression_ctrl.get_compression_state(),
            'nncf_metainfo': {
                # 'config': self._cfg,
                'nncf_config': self._cfg['nncf_config'],
                'nncf_compression_enabled': True,
            },
            'config': hyperparams_str,
            'labels': labels,
            'VERSION': 1
        }
        torch.save(modelinfo, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
