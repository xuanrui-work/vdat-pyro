from .base_runner import BaseRunner
from . import exception
from .options import RunnerOptions

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import torch.utils.tensorboard as tb

import numpy as np
from tqdm import tqdm

class EvalRunner(BaseRunner):
    def __init__(
        self,
        model: nn.Module|type,
        save_dir: str = '',
        progbar: bool = False,
        options: RunnerOptions|dict = None
    ):
        super().__init__(model, save_dir, options)

        self.progbar = progbar
    
    def set_loaders(self, loaders: list[DataLoader]):
        self.loaders = loaders
        self.len_loader = min(*[len(loader) for loader in loaders])
    
    def before_run(self):
        super().before_run()

        self.cum_stats = {}
        self.model.to(self.device)
    
    def before_epoch(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        for k, v in self.stats.items():
            if k not in self.cum_stats:
                self.cum_stats[k] = 0
            self.cum_stats[k] += self.batch_size * (v.item() if isinstance(v, torch.Tensor) else v)
    
    def after_epoch(self):
        pass

    def after_run(self):
        super().after_run()

        # average cumulated stats
        for k, v in self.cum_stats.items():
            self.cum_stats[k] = v / self.n_samples
        # save stats
        if self.save_dir:
            np.save(self.save_dir/'cum_stats.npy', self.cum_stats)

    def step(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def epoch(self):
        for i, ((xs, ys), (xt, yt)) in (
            tqdm(
                enumerate(zip(*self.loaders)),
                desc=f'{self.__class__.__name__}.epoch',
                total=self.len_loader,
                leave=False
            ) if self.progbar else enumerate(zip(*self.loaders))
        ):
            try:
                if xs.shape[0] != xt.shape[0]:
                    # truncate the larger batch
                    # m = min(xs.shape[0], xt.shape[0])
                    # xs, ys = xs[:m], ys[:m]
                    # xt, yt = xt[:m], yt[:m]
                    continue
                xs, ys = xs.to(self.device), ys.to(self.device)
                xt, yt = xt.to(self.device), yt.to(self.device)

                self.batch_size = xs.shape[0]
                self.batch = i

                self.before_step()
                self.stats = self.step(xs, ys, xt, yt)
                self.after_step()

                self.n_samples += self.batch_size
                self.n_step += 1
            
            except exception.SkipStep:
                continue
    
    def run(self):
        training = self.model.training
        self.model.eval()

        self.n_samples = 0
        self.n_step = 0
        try:
            with torch.no_grad():
                self.before_epoch()
                self.epoch()
                self.after_epoch()
        
        except exception.StopRun:
            pass
        finally:
            self.model.train(training)
    
    def execute(self, loaders: list[DataLoader] = None):
        if loaders is not None:
            self.set_loaders(loaders)
        elif not hasattr(self, 'loaders'):
            raise ValueError('no loaders set. call set_loaders() first')
        super().execute()
        return self.cum_stats.copy()

class TrainRunner(BaseRunner):
    def __init__(
        self,
        model: nn.Module|type,
        save_dir: str,
        eval_runner: EvalRunner = None,
        progbar: bool = False,
        options: RunnerOptions|dict = None
    ):
        super().__init__(model, save_dir, options)

        self.writer = tb.SummaryWriter(self.save_dir)

        self.progbar = progbar

        if eval_runner is None:
            eval_runner = EvalRunner(self.model, '', progbar, options)
            eval_runner.step = self.step
        self.eval_runner = eval_runner

        if self.options.val_metric[0] == '-':
            self.val_metric = self.options.val_metric[1:]
            self.neg_vm = True
        else:
            self.val_metric = self.options.val_metric
            self.neg_vm = False
    
    def set_loaders(
        self,
        train_loaders: DataLoader,
        val_loaders: DataLoader,
        test_loaders: DataLoader
    ):
        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.test_loaders = test_loaders
        self.len_loader = min(*[len(loader) for loader in train_loaders])
    
    def log_hparams(self, metric_dict: dict[str, float] = None):
        if metric_dict is None:
            metric_dict = {}
        metric_dict = {f'hparam/{k}': v for k, v in metric_dict.items()}

        # construct hparam_dict
        hd = self.options.to_dict()
        hd.update(self.options.model.to_dict())
        hd.update(self.options.oparams.to_dict())
        hd.update(self.options.hparams.to_dict())

        # convert/remove incompatible types
        allowable = (bool, str, float, int, torch.Tensor)
        for k, v in hd.copy().items():
            if not isinstance(v, allowable):
                if isinstance(v, dict):
                    hd.pop(k)
                else:
                    hd[k] = str(v)
        
        # write to tensorboard
        self.writer.add_hparams(hd, metric_dict, global_step=self.n_step)
    
    def visualize(self):
        pass

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))
    
    def before_run(self):
        super().before_run()

        self.n_epoch = 0
        self.n_step = 0
        self.best_val_metric = np.inf

        self.all_train_stats = {}
        self.all_val_stats = {}

        self.model.to(self.device)
        self.model.train()

        # log hparams
        self.writer.add_text('hparams', str(self.options))
        self.log_hparams()

    def before_epoch(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        train_stats = self.train_stats.copy()
        # logging
        for k, v in train_stats.items():
            self.writer.add_scalar(f'train/{k}', v, self.n_step)
        # update stats
        train_stats['step'] = self.n_step
        for k, v in train_stats.items():
            if k not in self.all_train_stats:
                self.all_train_stats[k] = []
            self.all_train_stats[k] += [v.item()] if isinstance(v, torch.Tensor) else [v]

    def after_epoch(self):
        if self.n_epoch % self.options.val_every == 0:
            val_stats = self.eval_runner.execute(self.val_loaders)
            self.val_stats = val_stats.copy()

            # logging
            for k, v in val_stats.items():
                self.writer.add_scalar(f'val/{k}', v, self.n_step)

            # update stats
            val_stats['step'] = self.n_step
            for k, v in val_stats.items():
                if k not in self.all_val_stats:
                    self.all_val_stats[k] = []
                self.all_val_stats[k] += [v.item()] if isinstance(v, torch.Tensor) else [v]
            
            # visualize
            self.visualize()
            
            # save model
            val_metric = val_stats[self.val_metric]
            if self.neg_vm:
                val_metric = -val_metric
            
            if val_metric < self.best_val_metric:
                self.best_val_metric = val_metric
                self.save_model(self.save_dir/'best_model.pth')
            self.save_model(self.save_dir/'last_model.pth')

            # save stats
            np.save(self.save_dir/'train_stats.npy', self.all_train_stats)
            np.save(self.save_dir/'val_stats.npy', self.all_val_stats)
    
    def after_run(self):
        super().after_run()

        # save last model
        self.save_model(self.save_dir/'last_model.pth')

        # evaluate on test set
        test_stats = self.eval_runner.execute(self.test_loaders)
        self.test_stats = test_stats.copy()

        # save stats
        np.save(self.save_dir/'test_stats.npy', test_stats)

        # stats on best model
        self.model.load_state_dict(torch.load(self.save_dir/'best_model.pth'))

        test_stats_bm = self.eval_runner.execute(self.test_loaders)
        self.test_stats_bm = test_stats_bm.copy()

        np.save(self.save_dir/'test_stats_bm.npy', test_stats_bm)

        # restore model
        self.load_model(self.save_dir/'last_model.pth')

        # log hparams on final val stats
        self.log_hparams(self.val_stats)

    def step(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def epoch(self):
        for i, ((xs, ys), (xt, yt)) in (
            tqdm(
                enumerate(zip(*self.train_loaders)),
                desc=f'{self.__class__.__name__}.epoch',
                total=self.len_loader,
                leave=False
            ) if self.progbar else enumerate(zip(*self.train_loaders))
        ):
            try:
                if xs.shape[0] != xt.shape[0]:
                    # truncate the larger batch
                    # m = min(xs.shape[0], xt.shape[0])
                    # xs, ys = xs[:m], ys[:m]
                    # xt, yt = xt[:m], yt[:m]
                    continue
                xs, ys = xs.to(self.device), ys.to(self.device)
                xt, yt = xt.to(self.device), yt.to(self.device)

                self.batch_size = xs.shape[0]
                self.batch = i

                self.before_step()
                self.train_stats = self.step(xs, ys, xt, yt)
                self.after_step()

                self.n_step += 1
            
            except exception.SkipStep:
                continue
    
    def run(self):
        try:
            for epoch in (
                tqdm(
                    range(self.options.num_epochs),
                    desc=f'{self.__class__.__name__}.run'
                ) if self.progbar else range(self.options.num_epochs)
            ):
                try:
                    self.before_epoch()
                    self.epoch()
                    self.after_epoch()
                    self.n_epoch += 1
                except exception.SkipEpoch:
                    continue
        
        except KeyboardInterrupt as err:
            pass
        
        except exception.StopRun:
            pass
    
    def execute(self):
        if not hasattr(self, 'train_loaders'):
            raise ValueError('no loaders set. call set_loaders() first')
        super().execute()
