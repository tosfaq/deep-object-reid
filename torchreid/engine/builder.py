from typing import Dict

from torchreid.engine import (ImageAMSoftmaxEngine, ImageContrastiveEngine,
                              ImageTripletEngine)
from torchreid.integration.nncf.engine import AccuracyAwareImageAMSoftmaxEngine
from torchreid.integration.nncf.utils import is_accuracy_aware_training


def get_params_for_image_am_softmax_engine(cfg) -> Dict[str, object]:
    image_am_softmax_engine_params = {
        'reg_cfg': cfg.reg,
        'metric_cfg': cfg.metric_losses,
        'use_gpu': cfg.use_gpu,
        'save_chkpt': cfg.model.save_chkpt,
        'train_patience': cfg.train.train_patience,
        'early_stoping': cfg.train.early_stoping,
        'lr_decay_factor': cfg.train.lr_decay_factor,
        'conf_penalty': cfg.loss.softmax.conf_penalty,
        'label_smooth': cfg.loss.softmax.label_smooth,
        'aug_type': cfg.loss.softmax.augmentations.aug_type,
        'aug_prob': cfg.loss.softmax.augmentations.aug_prob,
        'decay_power': cfg.loss.softmax.augmentations.fmix.decay_power,
        'alpha': cfg.loss.softmax.augmentations.alpha,
        'size': (cfg.data.height, cfg.data.width),
        'max_soft': cfg.loss.softmax.augmentations.fmix.max_soft,
        'reformulate': cfg.loss.softmax.augmentations.fmix.reformulate,
        'pr_product': cfg.loss.softmax.pr_product,
        'softmax_type': cfg.loss.name,
        'm': cfg.loss.softmax.m,
        's': cfg.loss.softmax.s,
        'compute_s': cfg.loss.softmax.compute_s,
        'margin_type': cfg.loss.softmax.margin_type,
        'end_s': cfg.loss.softmax.end_s,
        'duration_s': cfg.loss.softmax.duration_s,
        'skip_steps_s': cfg.loss.softmax.skip_steps_s,
        'enable_masks': cfg.data.enable_masks,
        'adaptive_margins': cfg.loss.softmax.adaptive_margins,
        'class_weighting': cfg.loss.softmax.class_weighting,
        'attr_cfg': cfg.attr_loss,
        'base_num_classes': cfg.loss.softmax.base_num_classes,
        'symmetric_ce': cfg.loss.softmax.symmetric_ce,
        'mix_weight': cfg.mixing_loss.enable * cfg.mixing_loss.weight,
        'enable_rsc': cfg.model.self_challenging_cfg.enable,
        'enable_sam': cfg.sam.enable,
        'use_ema_decay': cfg.train.ema.enable,
        'ema_decay': cfg.train.ema.ema_decay,
        'asl_gamma_neg': cfg.loss.asl.gamma_neg,
        'asl_gamma_pos': cfg.loss.asl.gamma_pos,
        'asl_p_m': cfg.loss.asl.p_m
    }
    return image_am_softmax_engine_params


def build_engine(cfg, datamanager, model, optimizer, scheduler,
                 should_freeze_aux_models=False,
                 nncf_metainfo=None,
                 initial_lr=None):
    if should_freeze_aux_models or nncf_metainfo:
        if cfg.loss.name not in ['softmax', 'am_softmax']:
            raise NotImplementedError('Freezing of aux models or NNCF compression are supported only for '
                                      'softmax and am_softmax losses for data.type = image')

    if cfg.loss.name in ['softmax', 'am_softmax', 'asl']:
        image_am_softmax_engine_params = get_params_for_image_am_softmax_engine(cfg)
        initial_lr = initial_lr if initial_lr else cfg.train.lr
        if nncf_metainfo is not None:
            if is_accuracy_aware_training(nncf_metainfo.nncf_config):
                # TODO:(kshpv) lets take *target_metric_name* from config?
                target_metric_name = 'top1'
                engine = AccuracyAwareImageAMSoftmaxEngine(
                    datamanager,
                    models=model,
                    optimizers=optimizer,
                    schedulers=scheduler,
                    should_freeze_aux_models=should_freeze_aux_models,
                    nncf_metainfo=nncf_metainfo,
                    initial_lr=initial_lr,
                    target_metric_name=target_metric_name,
                    **image_am_softmax_engine_params
                )
                return engine
        engine = ImageAMSoftmaxEngine(
            datamanager,
            models=model,
            optimizers=optimizer,
            schedulers=scheduler,
            should_freeze_aux_models=should_freeze_aux_models,
            nncf_metainfo=nncf_metainfo,
            initial_lr=initial_lr,
            **image_am_softmax_engine_params
        )
    elif cfg.loss.name == 'contrastive':
        engine = ImageContrastiveEngine(
            datamanager,
            model,
            optimizer=optimizer,
            reg_cfg=cfg.reg,
            scheduler=scheduler,
            use_gpu=cfg.use_gpu,
            s=cfg.loss.softmax.s,
            end_s=cfg.loss.softmax.end_s,
            duration_s=cfg.loss.softmax.duration_s,
            skip_steps_s=cfg.loss.softmax.skip_steps_s,
        )
    else:
        engine = ImageTripletEngine(
            datamanager,
            model,
            optimizer=optimizer,
            margin=cfg.loss.triplet.margin,
            weight_t=cfg.loss.triplet.weight_t,
            weight_x=cfg.loss.triplet.weight_x,
            scheduler=scheduler,
            use_gpu=cfg.use_gpu,
            label_smooth=cfg.loss.softmax.label_smooth,
            conf_penalty=cfg.loss.softmax.conf_penalty
        )

    return engine
