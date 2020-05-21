import os.path as osp
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, REMAINDER

import numpy as np
import torch
import torch.nn as nn

import torchreid
from torchreid.utils import load_pretrained_weights
from scripts.default_config import get_default_config, model_kwargs


def explore(model, max_scale=10.0):
    print('Norms:')
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

            if num_filters > filters.shape[1]:
                continue

            norms = np.sqrt(np.sum(np.square(filters), axis=-1))
            min_norm, max_norm = np.min(norms), np.max(norms)
            median_norm = np.median(norms)
            scale = max_norm / min_norm

            norm_filters = filters / norms.reshape([-1, 1])
            similarities = np.matmul(norm_filters, np.transpose(norm_filters))
            triu_values = similarities[np.triu_indices(similarities.shape[0], k=1)]
            median_sim = np.percentile(triu_values, 5)

            scales = max_norm / norms
            num_invalid = np.sum(scales > max_scale)
            if num_invalid >= 0:
                print('   - {} ({}): sim={:.3f} min={:.3f} median={:.3f} max={:.3f} scale={:.3f} invalid: {} / {}'
                      .format(name, kernel_type, median_sim,
                              min_norm, median_norm, max_norm, scale,
                              num_invalid, num_filters))


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
