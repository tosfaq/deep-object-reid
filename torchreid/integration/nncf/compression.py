import json
import numpy as np
import os
import sys
from collections import OrderedDict
from contextlib import contextmanager
from PIL import Image
from pprint import pformat

import torch


DEFAULT_COEFFICIENT_DECREASE_LR_FOR_NNCF = 0.035

@contextmanager
def nullcontext():
    """
    Context which does nothing
    """
    yield

def get_no_nncf_trace_context_manager():
    try:
        from nncf.dynamic_graph.context import \
            no_nncf_trace as original_no_nncf_trace
        return original_no_nncf_trace
    except ImportError:
        return nullcontext

def _get_nncf_metainfo_from_checkpoint(filename):
    if not filename:
        return None
    checkpoint = torch.load(filename, map_location='cpu')
    if not isinstance(checkpoint, dict):
        return None
    return checkpoint.get('nncf_metainfo', None)

def create_nncf_metainfo(nncf_compression_enabled, nncf_config):
    nncf_metainfo = {
            'nncf_compression_enabled': nncf_compression_enabled,
            'nncf_config': nncf_config,
        }
    return nncf_metainfo

def get_coeff_decrease_lr_for_nncf(nncf_external_config):
    if nncf_external_config and nncf_external_config.get('coeff_decrease_lr_for_nncf'):
        return nncf_external_config.get('coeff_decrease_lr_for_nncf')
    return DEFAULT_COEFFICIENT_DECREASE_LR_FOR_NNCF

def is_checkpoint_nncf(filename):
    nncf_metainfo = _get_nncf_metainfo_from_checkpoint(filename)
    if not nncf_metainfo:
        return False
    return nncf_metainfo.get('nncf_compression_enabled', False)

def _load_checkpoint_for_nncf(model, filename, map_location=None, strict=False):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    from nncf import load_state

    checkpoint = torch.load(filename, map_location=map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    _ = load_state(model, state_dict, strict)
    return checkpoint

def wrap_nncf_model(model, cfg, datamanager_for_init,
                    checkpoint_path=None):
    # Note that we require to import it here to avoid cyclic imports when import get_no_nncf_trace_context_manager
    # from mobilenetv3
    from torchreid.data.transforms import build_inference_transform

    if not (datamanager_for_init or checkpoint_path):
        raise RuntimeError(f'One of datamanager_for_init or checkpoint_path should be set: '
                           f'datamanager_for_init={datamanager_for_init} checkpoint_path={checkpoint_path}')

    import nncf
    from nncf import (NNCFConfig, create_compressed_model,
                      register_default_init_args)
    from nncf.initialization import InitializingDataLoader
    from nncf.dynamic_graph.input_wrapping import nncf_model_input
    from nncf.dynamic_graph.trace_tensor import TracedTensor

    # Taking the part, related to NNCF, from the model config.
    # Please, note that at the moment the part should be a dict of the following form:
    # ```
    # nncf = {
    #     'nncf_config': {
    #         # this is the NNCF config dict itself, placing in the model config file
    #         ....
    #     },
    #     coeff_decrease_lr_for_nncf: <float value> # this is a coefficient to decrease the LR for NNCF training
    # }
    # ```
    # -- in case if some section of this NNCF part is absent the default values will be used
    nncf_external_config = cfg.get('nncf')

    nncf_metainfo = _get_nncf_metainfo_from_checkpoint(checkpoint_path)
    if nncf_metainfo and nncf_metainfo.get('nncf_compression_enabled'):
        nncf_config_data = nncf_metainfo['nncf_config']
        datamanager_for_init = None
        print(f'Read NNCF metainfo from the checkpoint: nncf_metainfo=\n{pformat(nncf_metainfo)}')
    else:
        checkpoint_path = None # it is non-NNCF model

        if nncf_external_config:
            nncf_config_data = nncf_external_config.get('nncf_config')
            print(f'Read nncf config from the model config: nncf_config=\n{pformat(nncf_config_data)}')
        else:
            nncf_config_data = None

    if datamanager_for_init and checkpoint_path:
        raise RuntimeError(f'Only ONE of datamanager_for_init or checkpoint_path should be set: '
                           f'datamanager_for_init={datamanager_for_init} checkpoint_path={checkpoint_path}')
    if not nncf_config_data:
        print('Using the default NNCF int8 quantization config')
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

    # do it even if nncf_config_data is loaded from a checkpoint -- for the rare case when
    # the width and height of the model's input was changed in the config
    # and then finetuning of NNCF model is run
    h, w = cfg.data.height, cfg.data.width
    nncf_config_data.setdefault('input_info', {})
    nncf_config_data['input_info']['sample_size'] = [1, 3, h, w]

    nncf_config = NNCFConfig(nncf_config_data)
    print(f'nncf_config =\n{pformat(nncf_config)}')
    if not nncf_metainfo:
        nncf_metainfo = create_nncf_metainfo(nncf_compression_enabled=True,
                                             nncf_config=nncf_config_data)
    else:
        # update it just to be on the safe side
        nncf_metainfo['nncf_config'] = nncf_config_data

    class ReidInitializeDataLoader(InitializingDataLoader): #TODO: check is it correct
        def get_inputs(self, dataloader_output):
            # define own InitializingDataLoader class using approach like
            # parse_data_for_train and parse_data_for_eval in the class Engine
            # dataloader_output[0] should be image here
            args = (dataloader_output[0], )
            return args, {}

    cur_device = next(model.parameters()).device
    print(f'NNCF: cur_device = {cur_device}')

    if checkpoint_path is None:
        print('No NNCF checkpoint is provided -- register initialize data loader')
        train_loader = datamanager_for_init.train_loader
        wrapped_loader = ReidInitializeDataLoader(train_loader)
        nncf_config = register_default_init_args(nncf_config, wrapped_loader, device=cur_device)
        resuming_state_dict = None
    else:
        print(f'Loading NNCF model from {checkpoint_path}')
        resuming_state_dict = _load_checkpoint_for_nncf(model, checkpoint_path, map_location=cur_device)
        print(f'Loaded NNCF model from {checkpoint_path}')

    transform = build_inference_transform(
        cfg.data.height,
        cfg.data.width,
        norm_mean=cfg.data.norm_mean,
        norm_std=cfg.data.norm_std,
    )
    def random_image(height, width):
        input_size = (height, width, 3)
        img = np.random.rand(*input_size).astype(np.float32)
        img = np.uint8(img * 255)

        out_img = Image.fromarray(img)

        return out_img

    def dummy_forward(model):
        prev_training_state = model.training
        model.eval()
        input_img = random_image(cfg.data.height, cfg.data.width)
        input_blob = transform(input_img).unsqueeze(0)
        assert len(input_blob.size()) == 4
        input_blob = input_blob.to(device=cur_device)
        input_blob = nncf_model_input(input_blob)
        model(input_blob)
        model.train(prev_training_state)

    # TODO: think if this is required
    #       (NNCF has the default wrap_inputs builder)
    def wrap_inputs(args, kwargs):
        assert not kwargs
        assert len(args) == 1
        if isinstance(args[0], TracedTensor):
            print('wrap_inputs: do not wrap input TracedTensor')
            return args, {}
        return (nncf_model_input(args[0]), ), {}

    model.dummy_forward_fn = dummy_forward
    if 'log_dir' in nncf_config:
        os.makedirs(nncf_config['log_dir'], exist_ok=True)
    print(f'nncf_config["log_dir"] = {nncf_config["log_dir"]}')

    compression_ctrl, model = create_compressed_model(model,
                                                      nncf_config,
                                                      dummy_forward_fn=dummy_forward,
                                                      wrap_inputs_fn=wrap_inputs,
                                                      resuming_state_dict=resuming_state_dict)

    coeff_decrease_lr_for_nncf = get_coeff_decrease_lr_for_nncf(nncf_external_config)
    assert isinstance(coeff_decrease_lr_for_nncf, float)
    return compression_ctrl, model, nncf_metainfo, coeff_decrease_lr_for_nncf
