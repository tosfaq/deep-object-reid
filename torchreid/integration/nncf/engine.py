from torchreid.engine.image import ImageAMSoftmaxEngine


class AccuracyAwareImageAMSoftmaxEngine(ImageAMSoftmaxEngine):
    def __init__(self, datamanager, model, optimizer, reg_cfg, metric_cfg, scheduler=None, use_gpu=False,
                 save_chkpt=True, train_patience=10, early_stoping=False, lb_lr=1e-5, softmax_type='stock',
                 label_smooth=False, margin_type='cos', epsilon=0.1, aug_type=None, decay_power=3, alpha=1.,
                 size=(224, 224), max_soft=0.0, reformulate=False, aug_prob=1., conf_penalty=False, pr_product=False,
                 m=0.35, s=10, compute_s=False, end_s=None, duration_s=None, skip_steps_s=None, enable_masks=False,
                 adaptive_margins=False, class_weighting=False, attr_cfg=None, base_num_classes=-1, symmetric_ce=False,
                 mix_weight=1.0, enable_rsc=False, enable_sam=False, should_freeze_aux_models=False,
                 nncf_metainfo=None,
                 initial_lr=None, target_metric_name='top1'):
        super().__init__(datamanager, model, optimizer, reg_cfg, metric_cfg, scheduler, use_gpu, save_chkpt,
                         train_patience, early_stoping, lb_lr, softmax_type, label_smooth,
                         margin_type, epsilon, aug_type, decay_power, alpha, size, max_soft,
                         reformulate, aug_prob, conf_penalty, pr_product, m, s, compute_s, end_s,
                         duration_s, skip_steps_s, enable_masks, adaptive_margins, class_weighting,
                         attr_cfg, base_num_classes, symmetric_ce, mix_weight, enable_rsc, enable_sam,
                         should_freeze_aux_models,
                         nncf_metainfo,
                         initial_lr)

        self.target_metric_name = target_metric_name

    def run(self, compression_ctrl, nncf_config, **kwargs):
        from nncf.torch import AdaptiveCompressionTrainingLoop
        from nncf.torch import EarlyStoppingCompressionTrainingLoop

        self.train_data_loader = self.train_loader
        self.max_epoch = nncf_config.get('compression').get('accuracy_aware_training').get('maximal_total_epochs')
        acc_aware_training_loop = EarlyStoppingCompressionTrainingLoop(nncf_config,
                                                                       compression_ctrl)

        def _configure_optimizers_fn():
            return self.optims['model'], self.scheds['model']

        def dec_test(func):
            def inner(model, epoch):
                top1, top5, mAp = func(epoch)
                return top1

            return inner

        def dec_train(func):
            def inner(*args, **kwargs):
                self.epoch = kwargs['epoch']
                func()
                return

            return inner

        validate_f = dec_test(self.test)
        train_f = dec_train(self.train)

        model = acc_aware_training_loop.run(self.models['model'],
                                            train_epoch_fn=train_f,
                                            validate_fn=validate_f,
                                            configure_optimizers_fn=_configure_optimizers_fn)

        return model
