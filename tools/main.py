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
from torchreid.engine import build_engine
from torchreid.ops import DataParallel
from torchreid.utils import (Logger, check_isfile, collect_env_info,
                             compute_model_complexity,resume_from_checkpoint,
                             set_random_seed, load_pretrained_weights,
                             open_specified_layers)


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

def wrap_nncf_model(model, cfg, classification_classes_filter=None):
    import numpy as np
    import os
    from PIL import Image

    import nncf
    from nncf import (NNCFConfig, create_compressed_model,
                      register_default_init_args)
    from nncf.initialization import InitializingDataLoader
    from torchreid.data.transforms import build_inference_transform
    from nncf.dynamic_graph.input_wrapping import nncf_model_input

    h, w = cfg.data.height, cfg.data.width
    nncf_config_data = {
            "input_info": {
                "sample_size": [1, 3, h, w]
                },
            "compression": [
                {
                    "algorithm": "quantization",
                    "initializer": {
                        "range": {
                            "num_init_samples": 8192, # Number of samples from the training dataset to consume as sample model inputs for purposes of setting initial minimum and maximum quantization ranges
                            },
                        "batchnorm_adaptation": {
                            "num_bn_adaptation_samples": 8192, # Number of samples from the training dataset to pass through the model at initialization in order to update batchnorm statistics of the original model. The actual number of samples will be a closest multiple of the batch size.
                            #"num_bn_forget_samples": 1024, # Number of samples from the training dataset to pass through the model at initialization in order to erase batchnorm statistics of the original model (using large momentum value for rolling mean updates). The actual number of samples will be a closest multiple of the batch size.
                            }
                        }
                    }
                ],
            "log_dir": "."
            }
    nncf_config = NNCFConfig(nncf_config_data)
    print(f'nncf_config =\n{nncf_config}')

    print('before building datamanager for nncf initializing')
    datamanager = build_datamanager(cfg, classification_classes_filter)
    print('after building datamanager for nncf initializing')
    train_loader = datamanager.train_loader
    class ReidInitializeDataLoader(InitializingDataLoader): #TODO: check is it correct
        def get_inputs(self, dataloader_output):
            # define own InitializingDataLoader class using approach like
            # parse_data_for_train and parse_data_for_eval in the class Engine
            # dataloader_output[0] should be image here
            args = (dataloader_output[0], )
            return args, {}

    cur_device = next(model.parameters()).device
    print(f'cur_device = {cur_device}')

    # TODO: add `if not loading pretrained model` here
    wrapped_loader = ReidInitializeDataLoader(train_loader)
    nncf_config = register_default_init_args(nncf_config, wrapped_loader, device=cur_device)

    transform = build_inference_transform(
        cfg.data.height,
        cfg.data.width,
        norm_mean=cfg.data.norm_mean,
        norm_std=cfg.data.norm_std,
    )
    def random_image(height, width):
        if True:
            print(':::DEBUG: random_image call')
            import traceback
            traceback.print_stack(file=sys.stdout)
        input_size = (height, width, 3)
        img = np.random.rand(*input_size).astype(np.float32)
        img = np.uint8(img * 255)

        out_img = Image.fromarray(img)

        return out_img

    def dummy_forward(model):
        prev_training_state = model.training
        model.eval()
        input_img = random_image(cfg.data.height, cfg.data.width)
        input_blob = transform(input_img).unsqueeze(0)
        assert len(input_blob.size()) == 4
        input_blob = input_blob.to(device=cur_device)
        input_blob = nncf_model_input(input_blob)
        model(input_blob)
        model.train(prev_training_state)

    # TODO: think if this is required
    #       (NNCF has the default wrap_inputs builder)
    def wrap_inputs(args, kwargs):
        assert not kwargs
        assert len(args) == 1
        return (nncf_model_input(args[0]), ), {}

    model.dummy_forward_fn = dummy_forward
    if 'log_dir' in nncf_config:
        os.makedirs(nncf_config['log_dir'], exist_ok=True)
    print(f'nncf_config["log_dir"] = {nncf_config["log_dir"]}')

    resuming_state_dict = None #TODO
    compression_ctrl, model = create_compressed_model(model,
                                                      nncf_config,
                                                      dummy_forward_fn=dummy_forward,
                                                      wrap_inputs_fn=wrap_inputs,
                                                      resuming_state_dict=resuming_state_dict)
    return compression_ctrl, model


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

    should_freeze_aux_models = False
    if True: #TODO: add parameter to turn on nncf
        compression_ctrl, model = wrap_nncf_model(model, cfg, args.classes)
        should_freeze_aux_models = True

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

            if should_freeze_aux_models:
                aux_model = aux_model.eval()
                open_specified_layers(aux_model, [])

            models.append(aux_model)
            optimizers.append(aux_optimizer)
            schedulers.append(aux_scheduler)
    else:
        models, optimizers, schedulers = model, optimizer, scheduler

    print('Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type))
    engine = build_engine(cfg, datamanager, models, optimizers, schedulers,
                          should_freeze_aux_models=should_freeze_aux_models)

    engine.run(**engine_run_kwargs(cfg))


if __name__ == '__main__':
    main()
