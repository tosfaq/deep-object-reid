from torchreid.engine.image import ImageAMSoftmaxEngine


class AccuracyAwareImageAMSoftmaxEngine(ImageAMSoftmaxEngine):
    def __init__(self, datamanager, models, optimizers, reg_cfg, metric_cfg, *args, target_metric_name='top1', **kwargs):
        super().__init__(datamanager, models, optimizers, reg_cfg, metric_cfg, *args, **kwargs)
        self.target_metric_name = target_metric_name

    def run(self, compression_ctrl, nncf_config, **kwargs):
        from nncf.common.accuracy_aware_training import create_accuracy_aware_training_loop

        # Use only main model
        model = self.models[self.main_model_name]
        optimizer = self.optims[self.main_model_name]
        scheduler = self.scheds[self.main_model_name]

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
        self.max_epoch = nncf_config.get('accuracy_aware_training').get('params').get('maximal_total_epochs', 200)

        validate_fn = dec_test(self.test)
        train_fn = dec_train(self.train)

        acc_aware_training_loop = create_accuracy_aware_training_loop(nncf_config, compression_ctrl,
                                                                      lr_updates_needed=False)
        model = acc_aware_training_loop.run(model,
                                            train_epoch_fn=train_fn,
                                            validate_fn=validate_fn,
                                            configure_optimizers_fn=_configure_optimizers_fn)
        return model
