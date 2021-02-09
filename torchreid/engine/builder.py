from torchreid.engine import (ImageAMSoftmaxEngine, ImageContrastiveEngine,
                              ImageTripletEngine, VideoSoftmaxEngine,
                              VideoTripletEngine)


def build_engine(cfg, datamanager, model, optimizer, scheduler):
    if cfg.data.type == 'image':
        if cfg.loss.name in ['softmax', 'am_softmax']:
            softmax_type = 'stock' if cfg.loss.name == 'softmax' else 'am'
            engine = ImageAMSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                reg_cfg=cfg.reg,
                metric_cfg=cfg.metric_losses,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                save_chkpt = cfg.model.save_chkpt,
                train_patience = cfg.train.train_patience,
                early_stoping = cfg.train.early_stoping,
                lb_lr = cfg.train.min_lr,
                conf_penalty=cfg.loss.softmax.conf_penalty,
                label_smooth=cfg.loss.softmax.label_smooth,
                aug_type=cfg.loss.softmax.augmentations.aug_type,
                aug_prob=cfg.loss.softmax.augmentations.aug_prob,
                decay_power=cfg.loss.softmax.augmentations.fmix.decay_power,
                alpha=cfg.loss.softmax.augmentations.alpha,
                size=(cfg.data.height, cfg.data.width),
                max_soft=cfg.loss.softmax.augmentations.fmix.max_soft,
                reformulate=cfg.loss.softmax.augmentations.fmix.reformulate,
                pr_product=cfg.loss.softmax.pr_product,
                softmax_type=softmax_type,
                m=cfg.loss.softmax.m,
                s=cfg.loss.softmax.s,
                margin_type=cfg.loss.softmax.margin_type,
                end_s=cfg.loss.softmax.end_s,
                duration_s=cfg.loss.softmax.duration_s,
                skip_steps_s=cfg.loss.softmax.skip_steps_s,
                enable_masks=cfg.data.enable_masks,
                adaptive_margins=cfg.loss.softmax.adaptive_margins,
                class_weighting=cfg.loss.softmax.class_weighting,
                attr_cfg=cfg.attr_loss,
                base_num_classes=cfg.loss.softmax.base_num_classes,
                symmetric_ce=cfg.loss.softmax.symmetric_ce,
                mix_weight=cfg.mixing_loss.enable * cfg.mixing_loss.weight,
                enable_rsc=cfg.model.self_challenging_cfg.enable,
                enable_sam=cfg.sam.enable,
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
    else:
        if cfg.loss.name == 'softmax':
            engine = VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                conf_penalty=cfg.loss.softmax.conf_penalty,
                pooling_method=cfg.video.pooling_method
            )
        else:
            engine = VideoTripletEngine(
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
