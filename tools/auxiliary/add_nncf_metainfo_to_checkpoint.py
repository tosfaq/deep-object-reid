import argparse
import os

import torch

from scripts.default_config import get_default_config, merge_from_files_with_base
from torchreid.utils import save_checkpoint
from torchreid.integration.nncf.compression import get_default_nncf_compression_config

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='The script adds the default int8 quantization NNCF metainfo '
                                                 'to NNCF deep-object-reid checkpoints '
                                                 'that were trained when NNCF metainfo was not '
                                                 'stored in NNCF checkpoints')
    parser.add_argument('--config-file', type=str, required=True,
                        help='path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to the src checkpoint file')
    parser.add_argument('--dst-folder', type=str, required=True,
                        help='path to the dst folder to store dst checkpoint file')
    args = parser.parse_args()

    cfg = get_default_config()
    merge_from_files_with_base(cfg, args.config_file)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if not isinstance(checkpoint, dict):
        raise RuntimeError('Wrong format of checkpoint -- it is not the result of deep-object-reid training')
    if checkpoint.get('nncf_metainfo'):
        raise RuntimeError(f'Checkpoint {args.checkpoint} already has nncf_metainfo')

    if not os.path.isdir(args.dst_folder):
        raise RuntimeError(f'The dst folder {args.dst_folder} is NOT present')

    # default nncf config
    h, w = cfg.data.height, cfg.data.width
    nncf_config_data = get_default_nncf_compression_config(h, w)

    nncf_metainfo = {
            'nncf_compression_enabled': True,
            'nncf_config': nncf_config_data
    }
    checkpoint['nncf_metainfo'] = nncf_metainfo
    res_path = save_checkpoint(checkpoint, args.dst_folder)


if __name__ == '__main__':
    main()
