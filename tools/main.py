import argparse
import os.path as osp
import sys
import time
from pprint import pformat

from torch.utils.tensorboard import SummaryWriter
import torch
from scripts.default_config import (engine_run_kwargs, get_default_config,
                                    lr_finder_run_kwargs,
                                    lr_scheduler_kwargs, model_kwargs,
                                    optimizer_kwargs)
from scripts.script_utils import (build_base_argparser, reset_config,
                                  check_classes_consistency,
                                  build_datamanager, build_auxiliary_model)

import torchreid
from torchreid.engine import build_engine, get_initial_lr_from_checkpoint
from torchreid.ops import DataParallel
from torchreid.utils import (Logger, check_isfile, collect_env_info,
                             compute_model_complexity, resume_from_checkpoint,
                             set_random_seed, load_pretrained_weights)

from torchreid.integration.nncf.compression import wrap_nncf_model, is_checkpoint_nncf

def is_lr_set_from_opts(opts):
    # Note that opts here -- the args.opts from build_base_argparser
    key_names = opts[0::2]
    return ('train.lr' in key_names)

def main():
    parser = build_base_argparser()
    parser.add_argument('-e', '--auxiliary-models-cfg', type=str, nargs='*', default='',
                        help='path to extra config files')
    parser.add_argument('--split-models', action='store_true',
                        help='whether to split models on own gpu')

    parser.add_argument('--nncf', action='store_true',
                        help='If nncf compression should be used')
    parser.add_argument('--aux-config-opts', nargs='+', default=None,
                        help='Modify aux config options using the command-line')
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available() and args.gpu_num > 0
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    is_initial_lr_set_from_opts = is_lr_set_from_opts(args.opts)

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

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
            )
    elif cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)


    aux_lr = None # placeholder, needed for aux models, may be filled by nncf part below
    is_current_checkpoint_nncf = is_checkpoint_nncf(cfg.model.load_weights)
    if args.nncf or is_current_checkpoint_nncf:
        print(f'using NNCF')
        if cfg.model.resume:
            raise NotImplementedError('Resuming NNCF training not implemented yet')
        if not cfg.model.load_weights:
            raise RuntimeError('NNCF training should be started from a non-NNCF (or NNCF) pre-trained model')
        checkpoint_path = cfg.model.load_weights
        if not check_isfile(checkpoint_path):
            raise RuntimeError(f'Cannot load checkpoint from {checkpoint_path}')
        if is_current_checkpoint_nncf:
            # just skipping loading special datamanager
            datamanager_for_nncf = None
        else:
            print('before building datamanager for nncf initializing')
            datamanager_for_nncf = build_datamanager(cfg, args.classes)
            print('after building datamanager for nncf initializing')
            checkpoint_path = None

        compression_ctrl, model, nncf_metainfo, coeff_decrease_lr_for_nncf = \
                wrap_nncf_model(model, cfg, datamanager_for_nncf,
                                checkpoint_path=checkpoint_path)
        is_nncf_used = True
        should_freeze_aux_models = True
        print(f'should_freeze_aux_models = {should_freeze_aux_models}')
        print(f'Received from wrapping nncf_metainfo =\n{pformat(nncf_metainfo)}')
        if cfg.lr_finder.enable:
            print('Turn off LR finder -- it should not be used together with NNCF compression')
            cfg.lr_finder.enable = False

        # calculating initial LR for NNCF training
        if not is_initial_lr_set_from_opts:
            print('Try to calculate initial LR for NNCF')
            initial_lr_from_checkpoint = get_initial_lr_from_checkpoint(cfg.model.load_weights)
            print(f'initial_lr_from_checkpoint = {initial_lr_from_checkpoint}')
            if initial_lr_from_checkpoint is not None:
                print(f'coeff_decrease_lr_for_nncf = {coeff_decrease_lr_for_nncf}')
                aux_lr = initial_lr_from_checkpoint * coeff_decrease_lr_for_nncf
                cfg.train.lr = aux_lr
                print(f'calculated lr = {aux_lr}')
            else:
                print(f'The checkpoint does not contain initial lr -- will not calculate initial LR for NNCF, '
                      f'cfg.train.lr={cfg.train.lr}')
        else:
            print(f'Since initial LR is set from command line arguments, do not calculate initial LR for NNCF '
                  f'cfg.train.lr={cfg.train.lr}')

    else:
        is_nncf_used = False
        should_freeze_aux_models = False
        nncf_metainfo = None

    # creating optimizer and scheduler -- it should be done after NNCF part, since
    # NNCF could change some parameters
    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))

    if cfg.lr_finder.enable and cfg.lr_finder.mode == 'automatic' and not cfg.model.resume:
        scheduler = None
    else:
        scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(cfg))


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

    if cfg.lr_finder.enable and not cfg.test.evaluate and not cfg.model.resume:
        if enable_mutual_learning:
            print("Mutual learning is enabled. Learning rate will be estimated for the main model only.")

        # build new engine
        engine = build_engine(cfg, datamanager, model, optimizer, scheduler)
        aux_lr = engine.find_lr(**lr_finder_run_kwargs(cfg))

        print(f"Estimated learning rate: {aux_lr}")
        if cfg.lr_finder.stop_after:
            print("Finding learning rate finished. Terminate the training process")
            exit()

        # reload random seeds, optimizer with new lr and scheduler for it
        cfg.train.lr = aux_lr
        cfg.lr_finder.enable = False
        set_random_seed(cfg.train.seed)

        optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
        scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(cfg))

    if enable_mutual_learning:
        print('Enabled mutual learning between {} models.'.format(len(cfg.mutual_learning.aux_configs) + 1))

        models, optimizers, schedulers = [model], [optimizer], [scheduler]
        for config_file, device_ids in zip(cfg.mutual_learning.aux_configs, extra_device_ids):
            aux_model, aux_optimizer, aux_scheduler = build_auxiliary_model(
                config_file, num_train_classes, cfg.use_gpu, device_ids, lr=aux_lr,
                aux_config_opts=args.aux_config_opts
            )

            models.append(aux_model)
            optimizers.append(aux_optimizer)
            schedulers.append(aux_scheduler)
    else:
        models, optimizers, schedulers = model, optimizer, scheduler

    print('Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type))
    engine = build_engine(cfg, datamanager, models, optimizers, schedulers,
                          should_freeze_aux_models=should_freeze_aux_models,
                          nncf_metainfo=nncf_metainfo,
                          initial_lr=cfg.train.lr)

    log_dir = cfg.data.tb_log_dir if cfg.data.tb_log_dir else cfg.data.save_dir
    engine.run(**engine_run_kwargs(cfg), tb_writer=SummaryWriter(log_dir=log_dir))


if __name__ == '__main__':
    main()
