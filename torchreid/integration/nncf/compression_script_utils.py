from pprint import pformat

from torchreid.engine import get_initial_lr_from_checkpoint
from torchreid.utils import check_isfile, load_pretrained_weights

from scripts.script_utils import build_datamanager

from .compression import (is_checkpoint_nncf, wrap_nncf_model)

def calculate_lr_for_nncf_training(lr_from_cfg, checkpoint_path,
                                   coeff_decrease_lr_for_nncf,
                                   is_initial_lr_set_from_opts):
    if is_initial_lr_set_from_opts:
        print(f'Since initial LR is set from command line arguments, do not calculate initial LR for NNCF, '
              f'taking lr from cfg, lr={lr_from_cfg}')
        return lr_from_cfg

    initial_lr_from_checkpoint = get_initial_lr_from_checkpoint(checkpoint_path)
    print(f'initial_lr_from_checkpoint = {initial_lr_from_checkpoint}')
    if initial_lr_from_checkpoint is None:
        print(f'The checkpoint does not contain initial lr -- will not calculate initial LR for NNCF, '
              f'taking lr from cfg, lr={lr_from_cfg}')
        return lr_from_cfg

    if is_checkpoint_nncf(checkpoint_path):
        print('The loaded checkpoint was received from NNCF training')
        print('WARNING: Loading NNCF checkpoint for fine-tuning -- since LR is NOT set from command line arguments '
              'the initial LR stored in checkpoint will be used again')
        print('Please, set LR from command line arguments to avoid this behavior')
        lr = initial_lr_from_checkpoint
        print(f'lr = {lr}')
        return lr

    print('Try to calculate initial LR for NNCF')
    print(f'coeff_decrease_lr_for_nncf = {coeff_decrease_lr_for_nncf}')
    lr = initial_lr_from_checkpoint * coeff_decrease_lr_for_nncf
    print(f'calculated lr = {lr}')
    return lr

def make_nncf_changes_in_training(model, cfg, classes, is_initial_lr_set_from_opts):
    lr = None
    if cfg.model.resume:
        raise NotImplementedError('Resuming NNCF training is not implemented yet')
    if not cfg.model.load_weights:
        raise RuntimeError('NNCF training should be started from a non-NNCF or NNCF pre-trained model')
    checkpoint_path = cfg.model.load_weights
    if not check_isfile(checkpoint_path):
        raise RuntimeError(f'Cannot find checkpoint at {checkpoint_path}')

    is_curr_checkpoint_nncf = is_checkpoint_nncf(checkpoint_path)
    print(f'First stage of NNCF model wrapping -- loading weights from {checkpoint_path}')
    if is_curr_checkpoint_nncf:
        print('Note that it is an NNCF checkpoint, so warnings that some layers are discarded '
              'during loading due to unmatched keys will be printed -- it is normal for this case')

    load_pretrained_weights(model, checkpoint_path)

    if is_curr_checkpoint_nncf:
        print(f'Using NNCF checkpoint {checkpoint_path}')
        # just skipping loading special datamanager
        datamanager_for_nncf = None
        checkpoint_path_for_wrapping = checkpoint_path
    else:
        print('before building datamanager for nncf initializing')
        datamanager_for_nncf = build_datamanager(cfg, classes)
        print('after building datamanager for nncf initializing')
        checkpoint_path_for_wrapping = None

    compression_ctrl, model, nncf_metainfo, coeff_decrease_lr_for_nncf = \
            wrap_nncf_model(model, cfg, datamanager_for_nncf,
                            checkpoint_path=checkpoint_path_for_wrapping)
    print(f'Received from wrapping nncf_metainfo =\n{pformat(nncf_metainfo)}')
    if cfg.lr_finder.enable:
        print('Turn off LR finder -- it should not be used together with NNCF compression')
        cfg.lr_finder.enable = False

    # calculating initial LR for NNCF training
    lr = calculate_lr_for_nncf_training(cfg.train.lr, checkpoint_path,
                                        coeff_decrease_lr_for_nncf,
                                        is_initial_lr_set_from_opts)
    cfg.train.lr = lr
    return model, cfg, lr, nncf_metainfo

def make_nncf_changes_in_eval(model, cfg):
    print(f'using NNCF')
    checkpoint_path = cfg.model.load_weights
    datamanager_for_nncf = None
    compression_ctrl, model, _, _ = \
            wrap_nncf_model(model, cfg, datamanager_for_nncf,
                            checkpoint_path=checkpoint_path)
    return model, cfg
