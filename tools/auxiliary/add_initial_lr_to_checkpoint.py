# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import os

import torch

from scripts.default_config import get_default_config
from torchreid.utils import save_checkpoint

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='The script adds the initial LR value '
                                                 'to deep-object-reid checkpoints '
                                                 '-- it will allow using it for NNCF, '
                                                 'NNCF part will initialize its LR from the checkpoints`s LR')
    parser.add_argument('--lr', type=float, required=True,
                        help='path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to the src checkpoint file')
    parser.add_argument('--dst-folder', type=str, required=True,
                        help='path to the dst folder to store dst checkpoint file')
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if not isinstance(checkpoint, dict):
        raise RuntimeError('Wrong format of checkpoint -- it is not the result of deep-object-reid training')
    if checkpoint.get('initial_lr'):
        raise RuntimeError(f'Checkpoint {args.checkpoint} already has initial_lr')

    if not os.path.isdir(args.dst_folder):
        raise RuntimeError(f'The dst folder {args.dst_folder} is NOT present')

    checkpoint['initial_lr'] = float(args.lr)
    res_path = save_checkpoint(checkpoint, args.dst_folder)


if __name__ == '__main__':
    main()
