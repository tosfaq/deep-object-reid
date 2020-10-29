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
from PIL import Image

import numpy as np
import onnx
import torch
from torch.onnx.symbolic_registry import register_op
from torch.onnx.symbolic_helper import parse_args

from torchreid.models import build_model
from torchreid.utils import load_pretrained_weights
from torchreid.data.transforms import build_inference_transform
from scripts.default_config import get_default_config, model_kwargs


@parse_args('v', 'i', 'v', 'v', 'f', 'i')
def group_norm_symbolic(g, input, num_groups, weight, bias, eps, cudnn_enabled):
    from torch.onnx.symbolic_opset9 import reshape, mul, add, reshape_as

    channels_num = input.type().sizes()[1]

    if num_groups == channels_num:
        output = g.op('InstanceNormalization', input, weight, bias, epsilon_f=eps)
    else:
        # Reshape from [n, g * cg, h, w] to [1, n * g, cg * h, w].
        x = reshape(g, input, [0, num_groups, -1, 0])
        x = reshape(g, x, [1, -1, 0, 0])
        # Normalize channel-wise.
        x = g.op('MeanVarianceNormalization', x, axes_i=[2, 3])
        # Reshape back.
        x = reshape_as(g, x, input)
        # Apply affine transform.
        x = mul(g, x, reshape(g, weight, [1, channels_num, 1, 1]))
        output = add(g, x, reshape(g, bias, [1, channels_num, 1, 1]))

    return output


def parse_num_classes(source_datasets, classification=False, num_classes=None):
    if classification:
        assert num_classes is not None and len(num_classes) > 0

    if num_classes is not None and len(num_classes) > 0:
        return num_classes

    num_clustered = 0
    num_rest = 0
    for src in source_datasets:
        if isinstance(src, (tuple, list)):
            num_clustered += 1
        else:
            num_rest += 1

    total_num_sources = num_clustered + int(num_rest > 0)
    assert total_num_sources > 0

    return [0] * total_num_sources  # dummy number of classes


def random_image(height, width):
    input_size = (height, width, 3)
    img = np.random.rand(*input_size).astype(np.float32)
    img = np.uint8(img * 255)

    out_img = Image.fromarray(img)

    return out_img


def reset_config(cfg):
    cfg.model.download_weights = False


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str, default='',
                        help='Path to config file')
    parser.add_argument('--output-name', type=str, default='model',
                        help='Path to save ONNX model')
    parser.add_argument('--num-classes', type=int, nargs='+', default=None)
    parser.add_argument('--opset', type=int, default=9)
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='Verbose mode for onnx.export')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Modify config options using the command-line')
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    num_classes = parse_num_classes(cfg.data.sources, cfg.model.classification, args.num_classes)
    model = build_model(**model_kwargs(cfg, num_classes))
    load_pretrained_weights(model, cfg.model.load_weights)
    model.eval()

    transform = build_inference_transform(
        cfg.data.height,
        cfg.data.width,
        norm_mean=cfg.data.norm_mean,
        norm_std=cfg.data.norm_std,
    )

    input_img = random_image(cfg.data.height, cfg.data.width)
    input_blob = transform(input_img).unsqueeze(0)

    input_names = ['data']
    output_names = ['reid_embedding']
    dynamic_axes = {'data': {0: 'batch_size', 1: 'channels', 2: 'height', 3: 'width'},
                    'reid_embedding': {0: 'batch_size', 1: 'dim'}}

    output_file_path = args.output_name
    if not args.output_name.endswith('.onnx'):
        output_file_path += '.onnx'

    register_op("group_norm", group_norm_symbolic, "", args.opset)
    with torch.no_grad():
        torch.onnx.export(
            model,
            input_blob,
            output_file_path,
            verbose=args.verbose,
            export_params=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX
        )

    net_from_onnx = onnx.load(output_file_path)
    try:
        onnx.checker.check_model(net_from_onnx)
        print('ONNX check passed.')
    except onnx.onnx_cpp2py_export.checker.ValidationError as ex:
        print('ONNX check failed: {}.'.format(ex))


if __name__ == '__main__':
    main()
