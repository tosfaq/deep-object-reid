from functools import partial

import torch
from torch_lr_finder import LRFinder
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler, GridSampler
from addict import Dict
import time
import datetime

from .tools import get_model_attr

class LrFinder:
    def __init__(self,
        engine,
        mode='fast_ai',
        epochs_warmup=2,
        max_lr=0.03,
        min_lr=4e-3,
        step=0.001,
        num_epochs=3,
        path_to_savefig='',
        seed = 5,
        stop_callback=None,
        smooth_f=0.01,
        n_trials=30,
        **kwargs) -> None:
        r"""A  pipeline for learning rate search.

        Args:
            mode (str, optional): mode for learning rate finder, "fast_ai", "grid_search", "TPE".
                Default is "fast_ai".
            max_lr (float): upper bound for leaning rate
            min_lr (float): lower bound for leaning rate
            step (float, optional): number of step for learning rate searching space. Default is 1e-3
            num_epochs (int, optional): number of epochs to train for each learning rate. Default is 3
            pretrained (bool): whether or not the model is pretrained
            path_to_savefig (str): if path given save plot loss/lr (only for fast_ai mode). Default: ''
        """
        self.engine = engine
        main_model_name = engine.get_model_names(None)[0]
        self.model = engine.models[main_model_name]
        self.optimizer = engine.optims[main_model_name]
        self.model_device = next(self.model.parameters()).device
        self.mode = mode
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step = step
        self.n_trials = n_trials
        self.num_epochs = num_epochs
        # assert self.num_epochs > 3, "Number of epochs to find an optimal learning rate less than 3. It's pointless"
        self.path_to_savefig = path_to_savefig
        self.seed = seed
        self.stop_callback = stop_callback
        self.epochs_warmup = epochs_warmup
        self.enable_sam = engine.enable_sam
        self.smooth_f = smooth_f
        self.engine_cfg = Dict(min_lr=min_lr, max_lr=max_lr, mode=mode, step=step)
        self.samplers = {'grid_search': GridSampler(search_space={'lr': [0.005, 0.007, 0.01, 0.015, 0.02, 0.025, 0.03]}),
                            'TPE': TPESampler(n_startup_trials=5, seed=True)}

    def process(self):
        print('=> Start learning rate search. Mode: {}'.format(self.mode))
        if self.mode == 'fast_ai':
           lr = self.fast_ai()
           return lr

        assert self.mode in ['grid_search', 'TPE']
        lr = self.optuna_optim()
        return lr

    def fast_ai(self):
        wd = self.optimizer.param_groups[0]['weight_decay']
        criterion = self.engine.main_losses[0]
        if self.enable_sam:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.min_lr, weight_decay=wd)
        else:
            optimizer = self.optimizer

        if self.epochs_warmup != 0:
            get_model_attr(self.model, 'to')(self.model_device)
            print("Warmup the model's weights for {} epochs".format(self.epochs_warmup))
            self.engine.run(max_epoch=self.epochs_warmup, lr_finder=self.engine_cfg, stop_callback=self.stop_callback)
            print("Finished warmuping the model. Continue to find learning rate:")

        # run lr finder
        num_iter = len(self.engine.train_loader)
        lr_finder = LRFinder(self.model, optimizer, criterion, device=self.model_device)
        lr_finder.range_test(self.engine.train_loader, start_lr=self.min_lr, end_lr=self.max_lr,
                                smooth_f=self.smooth_f, num_iter=num_iter, step_mode='exp')
        ax, optim_lr = lr_finder.plot(suggest_lr=True)
        # save plot if needed
        if self.path_to_savefig:
            fig = ax.get_figure()
            fig.savefig(self.path_to_savefig)

        # reset weights and optimizer state
        if self.epochs_warmup != 0:
            self.engine.restore_model()
        else:
            lr_finder.reset()

        return optim_lr

    def optuna_optim(self):
        study = optuna.create_study(study_name='classification task', direction="maximize", sampler=self.samplers[self.mode])
        objective_partial = partial(self.engine.run, max_epoch=self.num_epochs, lr_finder=self.engine_cfg, start_eval=0, eval_freq=1,
                                stop_callback=self.stop_callback)
        try:
            start_time = time.time()
            study.optimize(objective_partial, n_trials=self.n_trials, timeout=None)
            elapsed = round(time.time() - start_time)
            print(f"--- learning rate estimation finished with elapsed time: {datetime.timedelta(seconds=elapsed)} ---")

        finally:
            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            print("Study statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))

            print("Best trial:")
            trial = study.best_trial

            print("  Value: ", trial.value)

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))

            return trial.params['lr']
