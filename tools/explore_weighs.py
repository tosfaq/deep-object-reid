import os.path as osp
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, REMAINDER

import numpy as np
import torch
import torch.nn as nn

import torchreid
from torchreid.utils import load_pretrained_weights
from scripts.default_config import get_default_config, model_kwargs


def explore(model, max_scale=5.0, max_similarity=0.2, sim_percentile=95):
    invalid_scales = []
    invalid_sim = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            weights = m.weight.detach().cpu().numpy()
            shape = weights.shape

            if len(shape) == 4:
                if shape[2] == shape[3] == 1:
                    kernel_type = '1x1'
                elif shape[1] == 1:
                    kernel_type = 'dw'
                else:
                    kernel_type = 'reg'
            else:
                kernel_type = 'fc'

            num_filters = shape[0]
            filters = weights.reshape([num_filters, -1])

            norms = np.sqrt(np.sum(np.square(filters), axis=-1))
            min_norm, max_norm = np.min(norms), np.max(norms)
            median_norm = np.median(norms)
            scale = max_norm / min_norm

            if num_filters <= filters.shape[1]:
                norm_filters = filters / norms.reshape([-1, 1])
                similarities = np.matmul(norm_filters, np.transpose(norm_filters))
                similarities = np.abs(similarities[np.triu_indices(similarities.shape[0], k=1)])

                num_invalid = np.sum(similarities > max_similarity)
                num_total = len(similarities)
                if num_invalid > 0:
                    sim = np.percentile(similarities, sim_percentile)
                    invalid_sim.append((name, kernel_type, sim, num_invalid, num_total, num_filters))

            scales = max_norm / norms
            num_invalid = np.sum(scales > max_scale)
            if num_invalid > 0:
                invalid_scales.append((name, kernel_type, median_norm, scale, num_invalid, num_filters))

    if len(invalid_scales) > 0:
        print('\nFound {} layers with invalid norm fraction (max/cur > {}):'
              .format(len(invalid_scales), max_scale))
        for name, kernel_type, median_norm, scale, num_invalid, num_filters in invalid_scales:
            print('   - {} ({}): {:.3f} (median={:.3f} invalid: {} / {})'
                  .format(name, kernel_type, scale, median_norm, num_invalid, num_filters))
    else:
        print('There are no layers with invalid norm.')

    if len(invalid_sim) > 0:
        print('\nFound {} layers with invalid similarity (value > {}):'
              .format(len(invalid_sim), max_similarity))
        for name, kernel_type, sim, num_invalid, num_total, num_filters in invalid_sim:
            print('   - {} ({}): {:.3f} (invalid: {} / {} size={})'
                  .format(name, kernel_type, sim, num_invalid, num_total, num_filters))
    else:
        print('There are no layers with invalid similarity.')


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('opts', default=None, nargs=REMAINDER)
    args = parser.parse_args()

    assert osp.exists(args.config)

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    model = torchreid.models.build_model(**model_kwargs(cfg, [0, 0]))
    load_pretrained_weights(model, cfg.model.load_weights)

    explore(model)


if __name__ == '__main__':
    main()
