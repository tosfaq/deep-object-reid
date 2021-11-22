from subprocess import run
import argparse
from pathlib import Path
import numpy as np
import re

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument( '--root', type=str, required=True,  help='path to folder with logs')
args = parser.parse_args()

datasets = dict(
         CIFAR100=dict(
            roots=['CIFAR100/train', 'CIFAR100/val'],
            names=['CIFAR100_train', 'CIFAR100_val'],
            types=['classification_image_folder', 'classification_image_folder'],
            sources='CIFAR100_train',
            targets='CIFAR100_val',
        ),
        flowers=dict(
            roots=['flowers/train.txt', 'flowers/val.txt'],
            names=['flowers_train', 'flowers_val'],
            types=['classification', 'classification'],
            sources='flowers_train',
            targets='flowers_val',
        ),
        cars=dict(
            roots=['cars/train.txt', 'cars/val.txt'],
            names=['cars_train', 'cars_val'],
            types=['classification', 'classification'],
            sources='cars_train',
            targets='cars_val',
        ),
        DTD=dict(
            roots=['DTD/train', 'DTD/val'],
            names=['DTD_train', 'DTD_val'],
            types=['classification_image_folder', 'classification_image_folder'],
            sources='DTD_train',
            targets='DTD_val',
        ),
        pets=dict(
            roots=['pets/train.txt', 'pets/val.txt'],
            names=['pets_train', 'pets_val'],
            types=['classification', 'classification'],
            sources='pets_train',
            targets='pets_val',
        ),
        birdsnap=dict(
            roots=['birdsnap/train.txt', 'birdsnap/val.txt'],
            names=['birdsnap_train', 'birdsnap_val'],
            types=['classification', 'classification'],
            sources='birdsnap_train',
            targets='birdsnap_val',
        ),
        caltech101=dict(
            roots=['caltech101/train.txt', 'caltech101/val.txt'],
            names=['caltech101_train', 'caltech101_val'],
            types=['classification', 'classification'],
            sources='caltech101_train',
            targets='caltech101_val',
        ),
        FOOD101=dict(
            roots=['FOOD101/train.txt', 'FOOD101/val.txt'],
            names=['FOOD101_train', 'FOOD101_val'],
            types=['classification', 'classification'],
            sources='FOOD101_train',
            targets='FOOD101_val',
        ),
        lg_chem=dict(
            roots=['lg_chem/train.txt', 'lg_chem/val.txt'],
            names=['lg_chem_train', 'lg_chem_val'],
            types=['classification', 'classification'],
            sources='lg_chem_train',
            targets='lg_chem_val',
        ),
        autism=dict(
            roots=['autism/train', 'autism/val'],
            names=['autism_train', 'autism_val'],
            types=['classification_image_folder', 'classification_image_folder'],
            sources='autism_train',
            targets='autism_val',
        ),
        attd_mi04_v4=dict(
            roots=['attd_mi04_v4/train.txt', 'attd_mi04_v4/val.txt'],
            names=['attd_mi04_v4_train', 'attd_mi04_v4_val'],
            types=['classification', 'classification'],
            sources='attd_mi04_v4_train',
            targets='attd_mi04_v4_val',
        ),
        attd_mi02_v3=dict(
            roots=['attd_mi02_v3/train.txt', 'attd_mi02_v3/val.txt'],
            names=['attd_mi02_v3_train', 'attd_mi02_v3_val'],
            types=['classification', 'classification'],
            sources='attd_mi02_v3_train',
            targets='attd_mi02_v3_val',
        )
    )

path_to_exp_folder = args.root
path_to_bash = str(Path.cwd() / 'tools/classification/parse_output.sh')
run(f'bash {path_to_bash} {path_to_exp_folder}', shell=True)
saver = dict()
path_to_file = f"{path_to_exp_folder}/combine_all.txt"
# parse output file from bash script
with open(path_to_file, 'r') as f:
    for line in f:
        if line.strip() in datasets.keys():
            next_dataset = line.strip()
            saver[next_dataset] = dict()
            continue
        else:
            for metric in ['mAP', 'Rank-1', 'Rank-5']:
                if metric in line.strip():
                    if not metric in saver[next_dataset]:
                        saver[next_dataset][metric] = []
                    pattern = re.search('\d+\.\d+', line.strip())
                    if pattern:
                        saver[next_dataset][metric].append(
                            float(pattern.group(0))
                        )

# dump in appropriate patern
names = ''
values = ''
with open(path_to_file, 'a') as f:
    for key in sorted(datasets.keys()):
        names += key + ' '
        if key in saver:
            best_top_1_idx = np.argmax(saver[key]['Rank-1'])
            top1 = str(saver[key]['Rank-1'][best_top_1_idx])
            mAP = str(saver[key]['mAP'][best_top_1_idx])
            top5 = str(saver[key]['Rank-5'][best_top_1_idx])
            values += mAP + ';' + top1 + ';' + top5 + ';'
        else:
            values += '-1;-1;-1;'

    f.write(f"\n{names}\n{values}")