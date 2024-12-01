from model.block_cnn import *

from runner.runner import TrainRunner, EvalRunner
from runner import exception
from runner.options import RunnerOptions

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Model(nn.Module):
    def __init__(
        self,
        in_shape=(3, 32, 32),
        n_cls=2,
        fc_hidden_dims=(1024, 512),
        hparams=None
    ):
        super().__init__()

        self.in_shape = in_shape
        self.n_cls = n_cls
        self.hparams = hparams

        self.classifier = Classifier(in_shape, n_cls, bn_mode='vanilla')

    def forward(self, x, d):
        d1 = self.classifier(x, d)
        outputs = {
            'x': x,
            'd1_lo': d1
        }
        if d == 'src':
            outputs = {k+'_A': v for k, v in outputs.items()}
        elif d == 'tgt':
            outputs = {k+'_B': v for k, v in outputs.items()}
        return outputs

class Evaluator(EvalRunner):
    def before_run(self):
        super().before_run()

        # loss functions
        self.loss_fn = nn.CrossEntropyLoss()

    def step(self, xs, ys, xt, yt):
        N = xs.shape[0]

        outputs = {}
        outputs.update(self.model(xs, 'src'))
        outputs.update(self.model(xt, 'tgt'))

        ds = torch.tensor([0]*N, device=xs.device)
        dt = torch.tensor([1]*N, device=xt.device)
        ld_s = self.loss_fn(outputs['d1_lo_A'], ds)
        ld_t = self.loss_fn(outputs['d1_lo_B'], dt)

        loss = (ld_s + ld_t) / 2

        acc_ds = (outputs['d1_lo_A'].argmax(-1) == 0).sum() / N
        acc_dt = (outputs['d1_lo_B'].argmax(-1) == 1).sum() / N
        acc = (acc_ds + acc_dt) / 2

        loss_dict = {
            'loss': loss,
            'ld_s': ld_s,
            'ld_t': ld_t,
            'acc': acc,
            'acc_ds': acc_ds,
            'acc_dt': acc_dt
        }
        return loss_dict

class Trainer(TrainRunner):
    def __init__(
        self,
        model: nn.Module,
        save_dir: str,
        progbar: bool = False,
        options: RunnerOptions|dict = None
    ):
        super().__init__(
            model,
            save_dir,
            Evaluator(
                model,
                '',
                progbar,
                options
            ),
            progbar,
            options
        )

    def before_run(self):
        super().before_run()

        # loss functions
        self.loss_fn = nn.CrossEntropyLoss()

        # setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.options.oparams.lr,
            betas=self.options.oparams.betas
        )

    def step(self, xs, ys, xt, yt):
        N = xs.shape[0]

        outputs = {}
        outputs.update(self.model(xs, 'src'))
        outputs.update(self.model(xt, 'tgt'))

        ds = torch.tensor([0]*N, device=xs.device)
        dt = torch.tensor([1]*N, device=xt.device)
        ld_s = self.loss_fn(outputs['d1_lo_A'], ds)
        ld_t = self.loss_fn(outputs['d1_lo_B'], dt)

        loss = (ld_s + ld_t) / 2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        acc_ds = (outputs['d1_lo_A'].argmax(-1) == 0).sum() / N
        acc_dt = (outputs['d1_lo_B'].argmax(-1) == 1).sum() / N
        acc = (acc_ds + acc_dt) / 2

        loss_dict = {
            'loss': loss,
            'ld_s': ld_s,
            'ld_t': ld_t,
            'acc': acc,
            'acc_ds': acc_ds,
            'acc_dt': acc_dt
        }
        return loss_dict
