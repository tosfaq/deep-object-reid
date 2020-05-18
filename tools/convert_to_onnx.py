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

import torch
import onnx
import numpy as np

from torchreid.models import build_model
from torchreid.utils import load_pretrained_weights
from torchreid.data.transforms import build_inference_transform
from scripts.default_config import get_default_config, model_kwargs


def random_image(height, width):
    input_size = (height, width, 3)
    img = np.random.rand(*input_size).astype(np.float32)
    img = np.uint8(img * 255)

    out_img = Image.fromarray(img)

    return out_img


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str, default='',
                        help='Path to config file')
    parser.add_argument('--output-name', type=str, default='model',
                        help='Path to save ONNX model')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='Verbose mode for onnx.export')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Modify config options using the command-line')
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    num_classes = [0, 0]  # dummy num classes for two-head architecture
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

    input_names = ['input']
    output_names = ['output']
    dynamic_axes = {'input': {0: 'batch_size', 1: 'channels', 2: 'height', 3: 'width'},
                    'output': {0: 'batch_size', 1: 'dim'}}
    output_file_path = '{}.onnx'.format(args.output_name)
    torch.onnx.export(
        model, input_blob, output_file_path, verbose=args.verbose, export_params=True,
        input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
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
