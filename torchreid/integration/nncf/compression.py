import os
from collections import OrderedDict
from pprint import pformat

import torch

from torchreid.utils.tools import random_image


class NNCFMetaInfo:
    def __init__(self, nncf_config=None, compression_enabled=True):
        self.nncf_config = nncf_config
        self.compression_enabled = compression_enabled

    @staticmethod
    def get_from_checkpoint(filename: str) -> 'NNCFMetaInfo':
        if not filename:
            return None
        checkpoint = torch.load(filename, map_location='cpu')
        if not isinstance(checkpoint, dict):
            return None
        if checkpoint.get('nncf_metainfo', None) is not None:
            return NNCFMetaInfo(checkpoint.get('nncf_metainfo', None))
        else:
            return None


def load_compression_state_from_checkpoint(model, filename, map_location=None, strict=False):
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


def wrap_nncf_model(model, cfg, nncf_config, datamanager_for_init,
                    checkpoint_path=None):
    if not (datamanager_for_init or checkpoint_path):
        raise RuntimeError(f'One of datamanager_for_init or checkpoint_path should be set: '
                           f'datamanager_for_init={datamanager_for_init} checkpoint_path={checkpoint_path}')

    if datamanager_for_init and checkpoint_path:
        raise RuntimeError(f'Only ONE of datamanager_for_init or checkpoint_path should be set: '
                           f'datamanager_for_init={datamanager_for_init} checkpoint_path={checkpoint_path}')

    from nncf.torch import create_compressed_model
    from nncf.torch.initialization import register_default_init_args

    from nncf.torch.dynamic_graph.io_handling import nncf_model_input
    from nncf.torch.dynamic_graph.trace_tensor import TracedTensor
    from nncf.torch.initialization import PTInitializingDataLoader

    nncf_metainfo = NNCFMetaInfo.get_from_checkpoint(checkpoint_path)
    if nncf_metainfo and nncf_metainfo.compression_enabled:
        print(f'Read NNCF metainfo with NNCF config from the checkpoint: nncf_metainfo=\n{pformat(nncf_metainfo)}')
    else:
        checkpoint_path = None  # it is non-NNCF model
        nncf_metainfo = NNCFMetaInfo(compression_enabled=True, nncf_config=nncf_config)

    cur_device = next(model.parameters()).device
    print(f'NNCF: cur_device = {cur_device}')

    if checkpoint_path is None:
        print('No NNCF checkpoint is provided -- register initialize data loader')

        class ReidInitializeDataLoader(PTInitializingDataLoader):
            def get_inputs(self, dataloader_output):
                # define own InitializingDataLoader class using approach like
                # parse_data_for_train and parse_data_for_eval in the class Engine
                # dataloader_output[0] should be image here
                args = (dataloader_output[0],)
                return args, {}

        def model_eval_fn(model):
            """
            Runs evaluation of the model on the validation set and
            returns the target metric value.
            Used to evaluate the original model before compression
            if NNCF-based accuracy-aware training is used.
            """
            from torchreid.metrics.classification import evaluate_classification

            if test_loader is None:
                raise RuntimeError('Cannot perform model evaluation on the validation '
                                   'dataset since the validation data loader was not passed '
                                   'to wrap_nncf_model')

            target_metric = 'top1'
            targets = list(test_loader.keys())
            use_gpu = True if 'cuda' == cur_device.type else False
            for dataset_name in targets:
                domain = 'source' if dataset_name in datamanager_for_init.sources else 'target'
                print('##### Evaluating {} ({}) #####'.format(dataset_name, domain))
                cmc, m_ap, norm_cm = evaluate_classification(test_loader[dataset_name]['query'], model,
                                                             use_gpu=use_gpu, topk=(1, 5))

            top1, top5 = cmc[0], cmc[1]
            if target_metric == 'top1':
                return top1
            else:
                raise NotImplementedError

        test_loader = datamanager_for_init.test_loader
        train_loader = datamanager_for_init.train_loader
        wrapped_loader = ReidInitializeDataLoader(train_loader)
        nncf_config = register_default_init_args(nncf_config, wrapped_loader,
                                                 model_eval_fn=model_eval_fn, device=cur_device)
        compression_state = None
    else:
        compression_state = load_compression_state_from_checkpoint(model, checkpoint_path, map_location=cur_device)
        print(f'Loaded NNCF model from {checkpoint_path}')

    def dummy_forward(model):
        from torchreid.data.transforms import build_inference_transform

        transform = build_inference_transform(
            cfg.data.height,
            cfg.data.width,
            norm_mean=cfg.data.norm_mean,
            norm_std=cfg.data.norm_std,
        )

        prev_training_state = model.training
        model.eval()
        input_img = random_image(cfg.data.height, cfg.data.width)
        input_blob = transform(input_img).unsqueeze(0)
        assert len(input_blob.size()) == 4
        input_blob = input_blob.to(device=cur_device)
        input_blob = nncf_model_input(input_blob)
        model(input_blob)
        model.train(prev_training_state)

    # TODO(lbeynens): improve this function to
    # avoid possible NNCF graph nodes duplication
    def wrap_inputs(args, kwargs):
        assert not kwargs
        assert len(args) == 1
        if isinstance(args[0], TracedTensor):
            print('wrap_inputs: do not wrap input TracedTensor')
            return args, {}
        return (nncf_model_input(args[0]),), {}

    model.dummy_forward_fn = dummy_forward
    if 'log_dir' in nncf_config:
        os.makedirs(nncf_config['log_dir'], exist_ok=True)
    print(f'nncf_config["log_dir"] = {nncf_config["log_dir"]}')

    compression_ctrl, model = create_compressed_model(model,
                                                      nncf_config,
                                                      dummy_forward_fn=dummy_forward,
                                                      wrap_inputs_fn=wrap_inputs,
                                                      compression_state=compression_state)

    return compression_ctrl, model, nncf_metainfo
