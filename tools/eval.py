"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import argparse
import os.path as osp
import time
import sys

import numpy as np
import torch
from PIL import Image
from scripts.default_config import get_default_config, model_kwargs, imagedata_kwargs

import torchreid
from torchreid.ops import DataParallel
from torchreid.data.datasets import init_image_dataset
from torchreid.data.transforms import build_inference_transform
from torchreid.models import build_model
from torchreid.utils import (Logger, set_random_seed, load_checkpoint, collect_env_info,
                             load_pretrained_weights, check_classes_consistency)
from torchreid.engine import build_engine


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


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str, default='',
                        help='path to config file')
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

    datamanager = torchreid.data.ImageDataManager(filter_classes=args.classes, **imagedata_kwargs(cfg))
    num_classes = len(datamanager.test_loader[cfg.data.targets[0]]['query'].dataset.classes)

    is_ie_model = cfg.model.load_weights.endswith('.xml')
    if not is_ie_model:
        model = torchreid.models.build_model(**model_kwargs(cfg, num_classes))
        load_pretrained_weights(model, cfg.model.load_weights)
    else:
        from torchreid.utils.ie_tools import VectorCNN
        from openvino.inference_engine import IECore
        cfg.test.batch_size = 1
        model = VectorCNN(IECore(), cfg.model.load_weights, 'CPU', switch_rb=True)
        model.classification_classes = []
        model.classification = True
        model.eval = lambda : None
        for name, dataloader in datamanager.test_loader.items():
            dataloader['query'].dataset.transform.transforms = \
                dataloader['query'].dataset.transform.transforms[:-2]

    if cfg.model.classification:
        classes_map = {v : k for k, v in enumerate(sorted(args.classes))} if args.classes else {}
        for name, dataloader in datamanager.test_loader.items():
            if not len(dataloader['query'].dataset.classes): # current text annotation doesn't contain classes names
                print(f'Warning: classes are not defined for validation dataset {name}')
            elif not len(model.classification_classes):
                    print(f'Warning: classes are not provided in the current snapshot. Consistency checks are skipped.')
            else:
                if not check_classes_consistency(model.classification_classes,
                                                 dataloader['query'].dataset.classes, strict=False):
                    raise ValueError('Inconsistent classes in evaluation dataset')
                if args.classes and not check_classes_consistency(classes_map,
                                                                model.classification_classes, strict=True):
                    raise ValueError('Classes provided via --classes should be the same as in the loaded model')

    if cfg.use_gpu and not is_ie_model:
        num_devices = min(torch.cuda.device_count(), args.gpu_num)
        main_device_ids = list(range(num_devices))
        model = DataParallel(model, device_ids=main_device_ids, output_device=0).cuda(main_device_ids[0])

    engine = build_engine(cfg, datamanager, model, None, None)
    engine.test(0,
                dist_metric=cfg.test.dist_metric,
                normalize_feature=cfg.test.normalize_feature,
                visrank=cfg.test.visrank,
                visrank_topk=cfg.test.visrank_topk,
                save_dir=cfg.data.save_dir,
                use_metric_cuhk03=cfg.cuhk03.use_metric_cuhk03,
                ranks=(1, 5, 10, 20),
                rerank=cfg.test.rerank)


if __name__ == '__main__':
    main()
