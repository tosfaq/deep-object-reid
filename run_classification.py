import argparse
import os
import re
import tempfile
from pathlib import Path
from subprocess import run

import numpy as np
from ruamel.yaml import YAML


def read_config(yaml: YAML, config_path: str):
    yaml.default_flow_style = True
    with open(config_path, 'r') as f:
        cfg = yaml.load(f)
    return cfg


def dump_config(yaml: YAML, config_path: str, cfg: dict):
    with open(config_path, 'w') as f:
        yaml.default_flow_style = True
        yaml.dump(cfg, f)


def compute_s(num_class: int):
    return max(np.sqrt(2) * np.log(num_class - 1), 3)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='/media/cluster_fs/datasets/classification',
        required=False,
        help='path to folder with datasets'
    )
    parser.add_argument(
        '--config', type=str, required=False, help='path to config file'
    )
    parser.add_argument('--gpu-num', type=int, default=1,
                        help='Number of GPUs for training. 0 is for CPU mode')
    args = parser.parse_args()
    print(args.gpu_num)
    path_to_main = './tools/main.py'
    data_root = args.data_root
    yaml = YAML()

    datasets = dict(
        CIFAR100=dict(
            resolution=(224, 224),
            epochs=35,
            roots=['/media/cluster_fs/datasets/classification/CIFAR100/train', '/media/cluster_fs/datasets/classification/CIFAR100/val'],
            names=['CIFAR100_train', 'CIFAR100_val'],
            types=['classification_image_folder', 'classification_image_folder'],
            sources='CIFAR100_train',
            targets='CIFAR100_val',
            batch_size=128,
            num_C=100
        ),
        SUN397=dict(
            resolution=(224, 224),
            epochs=60,
            roots=['/media/cluster_fs/datasets/classification/SUN397/train.txt', '/media/cluster_fs/datasets/classification/SUN397/val.txt'],
            names=['SUN397_train', 'SUN397_val'],
            types=['classification', 'classification'],
            sources='SUN397_train',
            targets='SUN397_val',
            batch_size=128,
            num_C=397
        ),
        flowers=dict(
            resolution=(224, 224),
            epochs=50,
            roots=['/media/cluster_fs/datasets/classification/flowers/train.txt', '/media/cluster_fs/datasets/classification/flowers/val.txt'],
            names=['flowers_train', 'flowers_val'],
            types=['classification', 'classification'],
            sources='flowers_train',
            targets='flowers_val',
            batch_size=128,
            num_C=102
        ),
        fashionMNIST=dict(
            resolution=(28, 28),
            epochs=35,
            roots=['/media/cluster_fs/datasets/classification/fashionMNIST/train', '/media/cluster_fs/datasets/classification/fashionMNIST/val'],
            names=['fashionMNIST_train', 'fashionMNIST_val'],
            types=['classification_image_folder', 'classification_image_folder'],
            sources='fashionMNIST_train',
            targets='fashionMNIST_val',
            batch_size=128,
            num_C=10
        ),
        SVHN=dict(
            resolution=(32, 32),
            epochs=50,
            roots=['/media/cluster_fs/datasets/classification/SVHN/train.txt', '/media/cluster_fs/datasets/classification/SVHN/val.txt'],
            names=['SVHN_train', 'SVHN_val'],
            types=['classification', 'classification'],
            sources='SVHN_train',
            targets='SVHN_val',
            batch_size=128,
            num_C=10
        ),
        cars=dict(
            resolution=(224, 224),
            epochs=110,
            roots=['/media/cluster_fs/datasets/classification/cars/train.txt', '/media/cluster_fs/datasets/classification/cars/val.txt'],
            names=['cars_train', 'cars_val'],
            types=['classification', 'classification'],
            sources='cars_train',
            targets='cars_val',
            batch_size=128,
            num_C=196
        ),
        DTD=dict(
            resolution=(224, 224),
            epochs=70,
            roots=['/media/cluster_fs/datasets/classification/DTD/train', '/media/cluster_fs/datasets/classification/DTD/val'],
            names=['DTD_train', 'DTD_val'],
            types=['classification_image_folder', 'classification_image_folder'],
            sources='DTD_train',
            targets='DTD_val',
            batch_size=128,
            num_C=47
        ),
        pets=dict(
            resolution=(224, 224),
            epochs=60,
            roots=['/media/cluster_fs/datasets/classification/pets/train.txt', '/media/cluster_fs/datasets/classification/pets/val.txt'],
            names=['pets_train', 'pets_val'],
            types=['classification', 'classification'],
            sources='pets_train',
            targets='pets_val',
            batch_size=128,
            num_C=37
        ),
        Xray=dict(
            resolution=(224, 224),
            epochs=70,
            roots=['/media/cluster_fs/datasets/classification/Xray/train', '/media/cluster_fs/datasets/classification/Xray/val'],
            names=['Xray_train', 'Xray_val'],
            types=['classification_image_folder', 'classification_image_folder'],
            sources='Xray_train',
            targets='Xray_val',
            batch_size=128,
            num_C=2
        ),
        birdsnap=dict(
            resolution=(224, 224),
            epochs=35,
            roots=['/media/cluster_fs/datasets/classification/birdsnap/train.txt', '/media/cluster_fs/datasets/classification/birdsnap/val.txt'],
            names=['birdsnap_train', 'birdsnap_val'],
            types=['classification', 'classification'],
            sources='birdsnap_train',
            targets='birdsnap_val',
            batch_size=128,
            num_C=500
        ),
        caltech101=dict(
            resolution=(224, 224),
            epochs=55,
            roots=['/media/cluster_fs/datasets/classification/caltech101/train.txt', '/media/cluster_fs/datasets/classification/caltech101/val.txt'],
            names=['caltech101_train', 'caltech101_val'],
            types=['classification', 'classification'],
            sources='caltech101_train',
            targets='caltech101_val',
            batch_size=128,
            num_C=101
        ),
        FOOD101=dict(
            resolution=(224, 224),
            epochs=35,
            roots=['/media/cluster_fs/datasets/classification/FOOD101/train.txt', '/media/cluster_fs/datasets/classification/FOOD101/val.txt'],
            names=['FOOD101_train', 'FOOD101_val'],
            types=['classification', 'classification'],
            sources='FOOD101_train',
            targets='FOOD101_val',
            batch_size=128,
            num_C=101
        )
    )

    path_to_base_cfg = args.config
    # to_skip = {'SUN397', 'birdsnap', 'CIFAR100', 'fashionMNIST', 'SVHN', 'cars', 'DTD', 'pets', 'Xray', 'caltech101', 'FOOD101', 'flowers'}
    # to_skip = {'SUN397', 'birdsnap', 'cars', 'DTD', 'pets', 'Xray', 'caltech101', 'FOOD101', 'flowers'}
    # to_skip = {'SUN397', 'Xray'}
    to_skip = {'Xray'}
    # to_skip = {'cars','flowers'}

    for key, params in datasets.items():
        if key in to_skip:
            continue
        cfg = read_config(yaml, path_to_base_cfg)
        # if key in {'CIFAR100', 'FOOD101', 'pets', 'SUN397'}:
        #     cfg['train']['lr'] = 0.01
        # elif key in {'DTD', 'Xray', 'birdsnap', 'caltech101', 'fashionMNIST'}:
        #     cfg['train']['lr'] = 0.02
        # else:
        #     cfg['train']['lr'] = 0.02
        path_to_exp_folder = cfg['data']['save_dir']
        # create new configuration file related to current dataset
        if cfg['loss']['name'] == "am_softmax":
            margin = compute_s(params['num_C'])
            cfg['loss']['softmax']['s'] = float(margin)

        if key in ['CIFAR100', 'SUN397']:
            cfg['train']['lr'] = 0.01
            cfg['lr_finder']['enable'] = False

        name_train = params['names'][0]
        name_val = params['names'][1]
        type_train = params['types'][0]
        type_val = params['types'][1]
        root_train = params['roots'][0]
        root_val = params['roots'][1]

        cfg['custom_datasets']['roots'] = [root_train, root_val]
        cfg['custom_datasets']['types'] = [type_train, type_val]
        cfg['custom_datasets']['names'] = [name_train, name_val]

        cfg['data']['height'] = params['resolution'][0]
        cfg['data']['width'] = params['resolution'][1]
        cfg['data']['save_dir'] = path_to_exp_folder + f"/{key}"

        source = params['sources']
        targets = params['targets']
        cfg['data']['sources'] = [source]
        cfg['data']['targets'] = [targets]
        # dump it
        # config_path = str(Path.cwd() / 'configs'/ 'classification' / f'{key}.yml')
        fd, tmp_path_to_cfg = tempfile.mkstemp()
        try:
            with os.fdopen(fd, 'w') as tmp:
                # do stuff with temp file
                yaml.default_flow_style = True
                yaml.dump(cfg, tmp)
                tmp.close()

            # run training
            run(
                f'python {str(path_to_main)}'
                f' --config {tmp_path_to_cfg}'
                f' --gpu-num {int(args.gpu_num)}',
                shell=True
            )
        finally:
            os.remove(tmp_path_to_cfg)

    # after training combine all outputs in one file
    # path_to_class_folder = f"outputs/classification_out/exp_{num_exp}"
    # path_to_bash = str(Path.cwd() / 'parse_output.sh')
    # run(f'bash {path_to_bash} {path_to_exp_folder}', shell=True)
    # saver = dict()
    # path_to_file = f"{path_to_exp_folder}/combine_all.txt"
    # # parse output file from bash script
    # with open(path_to_file, 'r') as f:
    #     for line in f:
    #         if line.strip() in datasets.keys():
    #             next_dataset = line.strip()
    #             saver[next_dataset] = dict()
    #             continue
    #         else:
    #             for metric in ['mAP', 'Rank-1', 'Rank-5']:
    #                 if line.strip().startswith(metric):
    #                     if not metric in saver[next_dataset]:
    #                         saver[next_dataset][metric] = []
    #                     pattern = re.search('\d+\.\d+', line.strip())
    #                     if pattern:
    #                         saver[next_dataset][metric].append(
    #                             float(pattern.group(0))
    #                         )

    # # dump in appropriate patern
    # names = ''
    # values = ''
    # with open(path_to_file, 'a') as f:
    #     for key in sorted(datasets.keys()):
    #         names += key + ' '
    #         if key in saver:
    #             best_top_1_idx = np.argmax(saver[key]['Rank-1'])
    #             top1 = str(saver[key]['Rank-1'][best_top_1_idx])
    #             mAP = str(saver[key]['mAP'][best_top_1_idx])
    #             top5 = str(saver[key]['Rank-5'][best_top_1_idx])
    #             snapshot = str(best_top_1_idx)
    #             values += mAP + ';' + top1 + ';' + top5 + ';' + snapshot + ';'
    #         else:
    #             values += '-1;-1;-1;-1;'

    #     f.write(f"\n{names}\n{values}")

if __name__ == "__main__":
    main()
