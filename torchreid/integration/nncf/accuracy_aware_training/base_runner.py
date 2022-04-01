# Copyright (C) 2022 Intel Corporation
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

from typing import Callable, Dict, List, Optional, Tuple, TypeVar
from nncf.api.compression import CompressionAlgorithmController
from nncf.common.accuracy_aware_training.runner import TrainingRunner

ModelType = TypeVar('ModelType')
OptimizerType = TypeVar('OptimizerType')
LRSchedulerType = TypeVar('LRSchedulerType')
TensorboardWriterType = TypeVar('TensorboardWriterType')


class BaseAccuracyAwareTrainingRunner(TrainingRunner):
    """
    The base accuracy-aware training Runner object,
    initialized with the default parameters unless specified in the config.
    """

    def __init__(self, accuracy_aware_params: Dict[str, object], verbose=True,
                 dump_checkpoints=True):
        self.maximal_relative_accuracy_drop = accuracy_aware_params.get('maximal_relative_accuracy_degradation', 1.0)
        self.maximal_absolute_accuracy_drop = accuracy_aware_params.get('maximal_absolute_accuracy_degradation')
        self.maximal_total_epochs = accuracy_aware_params.get('maximal_total_epochs', 10000)
        self.validate_every_n_epochs = accuracy_aware_params.get('validate_every_n_epochs', 1)

        self.verbose = verbose
        self.dump_checkpoints = dump_checkpoints

        self.accuracy_budget = None
        self.is_higher_metric_better = True
        self._compressed_training_history = []
        self._best_checkpoint = None

        self.training_epoch_count = 0
        self.cumulative_epoch_count = 0
        self.best_val_metric_value = 0
        self.loss = None

    def initialize_training_loop_fns(self, train_epoch_fn: Callable[[CompressionAlgorithmController, ModelType,
                                                                     Optional[OptimizerType],
                                                                     Optional[LRSchedulerType],
                                                                     Optional[int]], None],
                                     validate_fn: Callable[[ModelType, Optional[float]], float],
                                     configure_optimizers_fn: Callable[[], Tuple[OptimizerType, LRSchedulerType]],
                                     dump_checkpoint_fn: Callable[
                                         [ModelType, CompressionAlgorithmController, TrainingRunner, str], None],
                                     tensorboard_writer=None, log_dir=None):
        self._train_epoch_fn = train_epoch_fn
        self._validate_fn = validate_fn
        self._configure_optimizers_fn = configure_optimizers_fn
        self._dump_checkpoint_fn = dump_checkpoint_fn
        self._tensorboard_writer = tensorboard_writer
        self._log_dir = log_dir

    def calculate_minimal_tolerable_accuracy(self, uncompressed_model_accuracy: float):
        if self.maximal_absolute_accuracy_drop is not None:
            self.minimal_tolerable_accuracy = uncompressed_model_accuracy - self.maximal_absolute_accuracy_drop
        else:
            self.minimal_tolerable_accuracy = uncompressed_model_accuracy * \
                                              (1 - 0.01 * self.maximal_relative_accuracy_drop)


class BaseAdaptiveCompressionLevelTrainingRunner(BaseAccuracyAwareTrainingRunner):
    """
    The base adaptive compression level accuracy-aware training Runner object,
    initialized with the default parameters unless specified in the config.
    """

    def __init__(self, accuracy_aware_params: Dict[str, object], verbose=True,
                 minimal_compression_rate=0.05, maximal_compression_rate=0.95,
                 dump_checkpoints=True):
        super().__init__(accuracy_aware_params, verbose, dump_checkpoints)

        self.compression_rate_step = accuracy_aware_params.get('initial_compression_rate_step', 0.1)
        self.step_reduction_factor = accuracy_aware_params.get('compression_rate_step_reduction_factor', 0.5)
        self.minimal_compression_rate_step = accuracy_aware_params.get('minimal_compression_rate_step', 0.025)
        self.patience_epochs = accuracy_aware_params.get('patience_epochs')
        self.initial_training_phase_epochs = accuracy_aware_params.get('initial_training_phase_epochs')

        self.minimal_compression_rate = minimal_compression_rate
        self.maximal_compression_rate = maximal_compression_rate

        self._best_checkpoints = {}
        self._compression_rate_target = None
        self.adaptive_controller = None
        self.was_compression_increased_on_prev_step = None

    @property
    def compression_rate_target(self):
        if self._compression_rate_target is None:
            return self.adaptive_controller.compression_rate
        return self._compression_rate_target

    @compression_rate_target.setter
    def compression_rate_target(self, value):
        self._compression_rate_target = value

    def get_compression_rates_with_positive_acc_budget(self) -> List[float]:
        return [comp_rate for (comp_rate, acc_budget) in self._compressed_training_history if acc_budget >= 0]

    def get_compression_rates(self) -> List[float]:
        return [comp_rate for (comp_rate, _) in self._compressed_training_history]
