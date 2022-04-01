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

import os

from torchreid.utils import ModelEmaV2
from torchreid.integration.nncf.accuracy_aware_training.training_loop import (
    create_accuracy_aware_training_loop
)


def run_acc_aware_training_loop(engine, nncf_config, configure_optimizers_fn, stop_callback=None, perf_monitor=None):
    training_data = {}

    def dec_test(func):
        def inner(model, epoch):
            if engine.models[engine.main_model_name] != model:
                raise ValueError('Test function: invalid model')
            top1, training_data['should_save_ema_model'] = func(epoch)
            return top1
        return inner

    def dec_train(func):
        def inner(compression_controller, model, epoch, optimizer, lr_scheduler):
            if engine.compression_ctrl != compression_controller:
                raise ValueError('Train function: invalid compression controller')
            if engine.models[engine.main_model_name] != model:
                raise ValueError('Train function: invalid model')

            optimizers = optimizer if isinstance(optimizer, (tuple, list)) else [optimizer]
            lr_schedulers = lr_scheduler if isinstance(lr_scheduler, (tuple, list)) else [lr_scheduler]
            if len(engine.models) != len(optimizers) != len(lr_schedulers):
                raise ValueError('The number of optimizers and schedulers is not equal to the number of models.')

            for model_id, (optim, sched) in enumerate(zip(optimizers, lr_schedulers)):
                model_name = 'main_model' if model_id == 0 else f'aux_model_{model_id}'
                engine.optims[model_name] = optim
                engine.scheds[model_name] = sched

            engine.epoch = epoch
            return func(stop_callback=stop_callback, perf_monitor=perf_monitor)
        return inner

    def dec_update_learning_rate(func):
        def inner(lr_scheduler, epoch, accuracy, loss):
            lr_schedulers = lr_scheduler if isinstance(lr_scheduler, (tuple, list)) else [lr_scheduler]
            if len(engine.models) != len(lr_schedulers):
                raise ValueError('The number of schedulers is not equal to the number of models.')

            for model_id, sched in enumerate(lr_schedulers):
                model_name = 'main_model' if model_id == 0 else f'aux_model_{model_id}'
                engine.scheds[model_name] = sched

            engine.epoch = epoch
            target_metric = accuracy if engine.target_metric == 'test_acc' else loss
            func(output_avg_metric=target_metric)
        return inner

    def dec_early_stopping(func):
        def inner(accuracy):
            should_exit, _ = func(accuracy)
            return engine.early_stopping and should_exit
        return inner

    def dec_dump_checkpoint(func):
        def inner(model, compression_controller, _, log_dir):
            if engine.compression_ctrl != compression_controller:
                raise ValueError('Dump checkpoint function: invalid compression controller')
            if engine.models[engine.main_model_name] != model:
                raise ValueError('Dump checkpoint function: invalid model')
            func(0, log_dir, is_best=False, should_save_ema_model=training_data['should_save_ema_model'])
            return os.path.join(log_dir, 'latest.pth')
        return inner

    def reset_training():
        for model_name, model in engine.models.items():
            if model_name == engine.main_model_name and engine.use_ema_decay:
                engine.ema_model = ModelEmaV2(model, decay=engine.ema_model.decay)

        engine.best_metric = 0.0
        if hasattr(engine, 'prev_smooth_accuracy'):
            engine.prev_smooth_accuracy = 0.0

        return configure_optimizers_fn()

    acc_aware_training_loop = create_accuracy_aware_training_loop(nncf_config, engine.compression_ctrl,
                                                                  lr_updates_needed=False, dump_checkpoints=False)

    engine.train_data_loader = engine.train_loader
    engine.max_epoch = acc_aware_training_loop.runner.maximal_total_epochs

    validate_fn = dec_test(engine.test)
    train_fn = dec_train(engine.train)
    early_stopping_fn = dec_early_stopping(engine.exit_on_plateau_and_choose_best)
    dump_checkpoint_fn = dec_dump_checkpoint(engine.save_model)
    update_learning_rate_fn = dec_update_learning_rate(engine.update_lr)

    model = acc_aware_training_loop.run(engine.models[engine.main_model_name],
                                        train_epoch_fn=train_fn,
                                        validate_fn=validate_fn,
                                        configure_optimizers_fn=reset_training,
                                        early_stopping_fn=early_stopping_fn,
                                        dump_checkpoint_fn=dump_checkpoint_fn,
                                        update_learning_rate_fn=update_learning_rate_fn)
    return model
