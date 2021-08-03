from torchreid.engine.image import ImageAMSoftmaxEngine


class AccuracyAwareImageAMSoftmaxEngine(ImageAMSoftmaxEngine):
    def __init__(self, datamanager, models, optimizers, reg_cfg, metric_cfg, schedulers=None, use_gpu=False, save_chkpt=True,
                 train_patience=10, early_stoping = False, lr_decay_factor = 1000, softmax_type='stock', label_smooth=False,
                 margin_type='cos', epsilon=0.1, aug_type=None, decay_power=3, alpha=1., size=(224, 224), max_soft=0.0,
                 reformulate=False, aug_prob=1., conf_penalty=False, pr_product=False, m=0.35, s=10, compute_s=False, end_s=None,
                 duration_s=None, skip_steps_s=None, enable_masks=False, adaptive_margins=False, class_weighting=False,
                 attr_cfg=None, base_num_classes=-1, symmetric_ce=False, mix_weight=1.0, enable_rsc=False, enable_sam=False,
                 should_freeze_aux_models=False, nncf_metainfo=None, initial_lr=None, use_ema_decay=False, ema_decay=0.999, target_metric_name='top1'):
        super().__init__(datamanager, models, optimizers, reg_cfg, metric_cfg, schedulers, use_gpu, save_chkpt,
                         train_patience, early_stoping, lr_decay_factor, softmax_type, label_smooth,
                         margin_type, epsilon, aug_type, decay_power, alpha, size, max_soft,
                         reformulate, aug_prob, conf_penalty, pr_product, m, s, compute_s, end_s,
                         duration_s, skip_steps_s, enable_masks, adaptive_margins, class_weighting,
                         attr_cfg, base_num_classes, symmetric_ce, mix_weight, enable_rsc, enable_sam,
                         should_freeze_aux_models, nncf_metainfo, initial_lr, use_ema_decay, ema_decay)

        self.target_metric_name = target_metric_name

    def run(self, compression_ctrl, nncf_config, **kwargs):
        from nncf.common.accuracy_aware_training import create_accuracy_aware_training_loop

        def is_mutual_learning_enable():
            if len(self.models) > 1:
                return True
            elif len(self.models) == 1:
                return False
            else:
                raise RuntimeError('Incorrect number of models')

        if not is_mutual_learning_enable():
            model = self.models['model']
            optimizer = self.optims['model']
            scheduler = self.scheds['model']
        else:
            # Use only main model
            model = self.models['model_0']
            optimizer = self.optims['model_0']
            scheduler = self.scheds['model_0']

        def _configure_optimizers_fn():
            return optimizer, scheduler

        def dec_test(func):
            def inner(model, epoch):
                top1, top5, m_ap = func(epoch)
                if self.target_metric_name == 'top1':
                    return top1
                elif self.target_metric_name == 'mAP':
                    return m_ap
                else:
                    raise RuntimeError('Incorrect target metric')

            return inner

        def dec_train(func):
            def inner(*args, **kwargs):
                self.epoch = kwargs['epoch']
                func()
                return

            return inner

        self.train_data_loader = self.train_loader
        self.max_epoch = nncf_config.get('accuracy_aware_training').get('params').get('maximal_total_epochs')

        validate_fn = dec_test(self.test)
        train_fn = dec_train(self.train)

        acc_aware_training_loop = create_accuracy_aware_training_loop(nncf_config, compression_ctrl,
                                                                      lr_updates_needed=False)
        model = acc_aware_training_loop.run(model,
                                            train_epoch_fn=train_fn,
                                            validate_fn=validate_fn,
                                            configure_optimizers_fn=_configure_optimizers_fn)
        return model
