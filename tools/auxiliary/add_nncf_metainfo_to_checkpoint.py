import argparse
import os

import torch

from scripts.default_config import get_default_config
from torchreid.utils import save_checkpoint

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
    cfg.merge_from_file(args.config_file)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if not isinstance(checkpoint, dict):
        raise RuntimeError('Wrong format of checkpoint -- it is not the result of deep-object-reid training')
    if checkpoint.get('nncf_metainfo'):
        raise RuntimeError(f'Checkpoint {args.checkpoint} already has nncf_metainfo')

    if not os.path.isdir(args.dst_folder):
        raise RuntimeError(f'The dst folder {args.dst_folder} is NOT present')

    # default nncf config
    nncf_config_data = {
#            "input_info": {
#                "sample_size": [1, 3, h, w]
#                },
        "compression": [
            {
                "algorithm": "quantization",
                "initializer": {
                    "range": {
                        "num_init_samples": 8192, # Number of samples from the training dataset
                                                  # to consume as sample model inputs for purposes of setting initial
                                                  # minimum and maximum quantization ranges
                        },
                    "batchnorm_adaptation": {
                        "num_bn_adaptation_samples": 8192, # Number of samples from the training
                                                           # dataset to pass through the model at initialization in order to update
                                                           # batchnorm statistics of the original model. The actual number of samples
                                                           # will be a closest multiple of the batch size.
                        #"num_bn_forget_samples": 1024, # Number of samples from the training
                                                        # dataset to pass through the model at initialization in order to erase
                                                        # batchnorm statistics of the original model (using large momentum value
                                                        # for rolling mean updates). The actual number of samples will be a
                                                        # closest multiple of the batch size.
                        }
                    }
                }
            ],
        "log_dir": "."
    }
    h, w = cfg.data.height, cfg.data.width
    nncf_config_data.setdefault('input_info', {})
    nncf_config_data['input_info']['sample_size'] = [1, 3, h, w]

    nncf_metainfo = {
            'nncf_compression_enabled': True,
            'nncf_config': nncf_config_data
        }
    checkpoint['nncf_metainfo'] = nncf_metainfo
    res_path = save_checkpoint(checkpoint, args.dst_folder)


if __name__ == '__main__':
    main()
