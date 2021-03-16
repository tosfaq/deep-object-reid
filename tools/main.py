import argparse
import os.path as osp
import sys
import time

import torch
from scripts.default_config import (engine_run_kwargs, get_default_config,
                                    imagedata_kwargs, lr_finder_run_kwargs,
                                    lr_scheduler_kwargs, model_kwargs,
                                    optimizer_kwargs, videodata_kwargs)

import torchreid
from torchreid.engine import build_engine, EpochIntervalToValue
from torchreid.ops import DataParallel
from torchreid.utils import (Logger, check_isfile, collect_env_info,
                             compute_model_complexity,resume_from_checkpoint,
                             set_random_seed, load_pretrained_weights,
                             open_specified_layers)

from torchreid.integration.nncf.compression import wrap_nncf_model

def build_datamanager(cfg, classification_classes_filter=None):
    if cfg.data.type == 'image':
        return torchreid.data.ImageDataManager(filter_classes=classification_classes_filter, **imagedata_kwargs(cfg))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))

def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root

    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets

    if args.custom_roots:
        cfg.custom_datasets.roots = args.custom_roots
    if args.custom_types:
        cfg.custom_datasets.types = args.custom_types
    if args.custom_names:
        cfg.custom_datasets.names = args.custom_names

    if args.auxiliary_models_cfg:
        cfg.mutual_learning.aux_configs = args.auxiliary_models_cfg

def build_auxiliary_model(config_file, num_classes, use_gpu, device_ids=None, weights=None):
    cfg = get_default_config()
    cfg.use_gpu = use_gpu
    cfg.merge_from_file(config_file)

    print('\nShow auxiliary configuration\n{}\n'.format(cfg))

    model = torchreid.models.build_model(**model_kwargs(cfg, num_classes))

    if (weights is not None) and (check_isfile(weights)):
        load_pretrained_weights(model, weights)

    if cfg.use_gpu:
        assert device_ids is not None

        if len(device_ids) > 1:
            model = DataParallel(model, device_ids=device_ids, output_device=0).cuda(device_ids[0])
        else:
            model = model.cuda(device_ids[0])

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(cfg))
    return model, optimizer, scheduler

def check_classes_consistency(ref_classes, probe_classes, strict=False):
    if strict:
        if len(ref_classes) != len(probe_classes):
            return False
        return sorted(probe_classes.keys()) == sorted(ref_classes.keys())
    else:
        if len(ref_classes) > len(probe_classes):
            return False
        probe_names = probe_classes.keys()
        for cl in ref_classes.keys():
            if cl not in probe_names:
                return False
    return True

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str, default='',
                        help='path to config file')
    parser.add_argument('-e', '--auxiliary-models-cfg', type=str, nargs='*', default='',
                        help='path to extra config files')
    parser.add_argument('-w', '--extra-weights', type=str, nargs='*', default='',
                        help='path to extra model weights')
    parser.add_argument('--split-models', action='store_true',
                        help='whether to split models on own gpu')
    parser.add_argument('-s', '--sources', type=str, nargs='+',
                        help='source datasets (delimited by space)')
    parser.add_argument('-t', '--targets', type=str, nargs='+',
                        help='target datasets (delimited by space)')
    parser.add_argument('--root', type=str, default='',
                        help='path to data root')
    parser.add_argument('--classes', type=str, nargs='+',
                        help='name of classes in classification dataset')
    parser.add_argument('--custom-roots', type=str, nargs='+',
                        help='types or paths to annotation of custom datasets (delimited by space)')
    parser.add_argument('--custom-types', type=str, nargs='+',
                        help='path of custom datasets (delimited by space)')
    parser.add_argument('--custom-names', type=str, nargs='+',
                        help='names of custom datasets (delimited by space)')
    parser.add_argument('--gpu-num', type=int, default=1,
                        help='Number of GPUs for training. 0 is for CPU mode')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Modify config options using the command-line')
    parser.add_argument('--should_freeze_aux_models_for_nncf', default='True', choices=['True', 'False'],
                        help='If aux models should be frozen for NNCF ("True" or "False")')
    parser.add_argument('--turn_off_mutual_learning_up_to_epoch', default=None, type=int,
                        help='If set, mutual learning will be turned up to the pointed epoch (inluding the epoch)')
    parser.add_argument('--freeze_aux_model_up_to_epoch', default=None, type=int,
                        help='If set, auxiliary models will be frozen up to the pointed epoch (inluding the epoch)')
    parser.add_argument('--nncf', nargs='?', const=True, default=None,
                        help='If nncf compression should be used')
    parser.add_argument('--no_nncf', action='store_true',
                        help='If nncf compression should NOT be used')
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available() and args.gpu_num > 0
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)

    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    enable_mutual_learning = len(cfg.mutual_learning.aux_configs) > 0

    datamanager = build_datamanager(cfg, args.classes)
    num_train_classes = datamanager.num_train_pids

    print('Building main model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(**model_kwargs(cfg, num_train_classes))
    num_params, flops = compute_model_complexity(model, (1, 3, cfg.data.height, cfg.data.width))
    print('Main model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    should_freeze_aux_models_for_nncf = (args.should_freeze_aux_models_for_nncf == 'True')
    should_freeze_aux_models = False

    should_use_nncf = None
    if args.nncf and args.no_nncf:
        raise RuntimeError('Both --nncf and --no_nncf are set')
    if args.nncf:
        should_use_nncf = True
    if args.no_nncf:
        should_use_nncf = False
    if not isinstance(should_use_nncf, bool):
        # TODO(lbeynens): REMOVE THIS, it is a DEBUG insertion!
        print('ATTENTION: during debug NNCF features, the default action for training without options'
              ' --nncf and --no_nncf is USE NNCF -- it is TEMPORARY and will be turned off!')
        should_use_nncf = True

    if should_use_nncf:
        nncf_config_path = args.nncf if isinstance(args.nncf, str) else None
        print(f'should_freeze_aux_models_for_nncf = {should_freeze_aux_models_for_nncf}')
        print('before building datamanager for nncf initializing')
        datamanager_for_nncf = build_datamanager(cfg, args.classes)
        print('after building datamanager for nncf initializing')
        compression_ctrl, model = wrap_nncf_model(model, cfg, datamanager_for_nncf, nncf_config_path)
        should_freeze_aux_models = should_freeze_aux_models_for_nncf

    if args.freeze_aux_model_up_to_epoch is not None:
        should_freeze_aux_models = True
        epoch_interval_for_aux_model_freeze = EpochIntervalToValue(first=None,
                                                                   last=int(args.freeze_aux_model_up_to_epoch),
                                                                   value_inside=True,
                                                                   value_outside=False)
    else:
        epoch_interval_for_aux_model_freeze = None
    print(f'epoch_interval_for_aux_model_freeze = {epoch_interval_for_aux_model_freeze}')

    if args.turn_off_mutual_learning_up_to_epoch is not None:
        epoch_interval_for_turn_off_mutual_learning = EpochIntervalToValue(first=None,
                                                                           last=int(args.turn_off_mutual_learning_up_to_epoch),
                                                                           value_inside=True,
                                                                           value_outside=False)
    else:
        epoch_interval_for_turn_off_mutual_learning = None
    print(f'epoch_interval_for_turn_off_mutual_learning = {epoch_interval_for_turn_off_mutual_learning}')

    if cfg.model.classification:
        classes_map = {v : k for k, v in enumerate(sorted(args.classes))} if args.classes else {}
        if cfg.test.evaluate:
            for name, dataloader in datamanager.test_loader.items():
                if not len(dataloader['query'].dataset.classes): # current text annotation doesn't contain classes names
                    print(f'Warning: classes are not defined for validation dataset {name}')
                    continue
                if not len(model.classification_classes):
                    print(f'Warning: classes are not provided in the current snapshot. Consistency checks are skipped.')
                    continue
                if not check_classes_consistency(model.classification_classes,
                                                 dataloader['query'].dataset.classes, strict=False):
                    raise ValueError('Inconsistent classes in evaluation dataset')
                if args.classes and not check_classes_consistency(classes_map,
                                                                  model.classification_classes, strict=True):
                    raise ValueError('Classes provided via --classes should be the same as in the loaded model')
        elif args.classes:
            if not check_classes_consistency(classes_map,
                                             datamanager.train_loader.dataset.classes, strict=True):
                raise ValueError('Inconsistent classes in training dataset')

    if cfg.use_gpu:
        num_devices = min(torch.cuda.device_count(), args.gpu_num)
        if enable_mutual_learning and args.split_models:
            num_models = len(cfg.mutual_learning.aux_configs) + 1
            assert num_devices >= num_models
            assert num_devices % num_models == 0

            num_devices_per_model = num_devices // num_models
            device_splits = []
            for model_id in range(num_models):
                device_splits.append([
                    model_id * num_devices_per_model + i
                    for i in range(num_devices_per_model)
                ])

            main_device_ids = device_splits[0]
            extra_device_ids = device_splits[1:]
        else:
            main_device_ids = list(range(num_devices))
            extra_device_ids = [main_device_ids for _ in range(len(cfg.mutual_learning.aux_configs))]

        if num_devices > 1:
            model = DataParallel(model, device_ids=main_device_ids, output_device=0).cuda(main_device_ids[0])
        else:
            model = model.cuda(main_device_ids[0])
    else:
        extra_device_ids = [None for _ in range(len(cfg.mutual_learning.aux_configs))]

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))

    if cfg.lr_finder.enable and cfg.lr_finder.mode == 'automatic' and not cfg.model.resume:
        scheduler = None
    else:
        scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(cfg))

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )

    if cfg.lr_finder.enable and not cfg.test.evaluate and not cfg.model.resume:
        if enable_mutual_learning:
            print("Mutual learning is enabled. Learning rate will be estimated for the main model only.")

        # build new engine
        engine = build_engine(cfg, datamanager, model, optimizer, scheduler)
        lr = engine.find_lr(**lr_finder_run_kwargs(cfg))

        print(f"Estimated learning rate: {lr}")
        if cfg.lr_finder.stop_after:
            print("Finding learning rate finished. Terminate the training process")
            exit()

        # reload random seeds, opimizer with new lr and scheduler for it
        cfg.train.lr = lr
        cfg.lr_finder.enable = False
        set_random_seed(cfg.train.seed)

        optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
        scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(cfg))

    if enable_mutual_learning:
        print('Enabled mutual learning between {} models.'.format(len(cfg.mutual_learning.aux_configs) + 1))

        if len(args.extra_weights) > 0:
            assert len(args.extra_weights) == len(cfg.mutual_learning.aux_configs)
            weights = args.extra_weights
        else:
            weights = [None] * len(cfg.mutual_learning.aux_configs)

        models, optimizers, schedulers = [model], [optimizer], [scheduler]
        for config_file, model_weights, device_ids in zip(cfg.mutual_learning.aux_configs, weights, extra_device_ids):
            aux_model, aux_optimizer, aux_scheduler = build_auxiliary_model(
                config_file, num_train_classes, cfg.use_gpu, device_ids, model_weights
            )

            models.append(aux_model)
            optimizers.append(aux_optimizer)
            schedulers.append(aux_scheduler)
    else:
        models, optimizers, schedulers = model, optimizer, scheduler

    print('Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type))
    engine = build_engine(cfg, datamanager, models, optimizers, schedulers,
                          should_freeze_aux_models=should_freeze_aux_models,
                          epoch_interval_for_aux_model_freeze=epoch_interval_for_aux_model_freeze,
                          epoch_interval_for_turn_off_mutual_learning=epoch_interval_for_turn_off_mutual_learning)

    engine.run(**engine_run_kwargs(cfg))


if __name__ == '__main__':
    main()
