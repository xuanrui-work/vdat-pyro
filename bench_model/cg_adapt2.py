from .cg_adapt1 import CycleGAN as CycleGAN
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
        n_cls=10,
        fc_hidden_dims=(1024, 512),
        hparams=None
    ):
        super().__init__()

        self.in_shape = in_shape
        self.n_cls = n_cls
        self.hparams = hparams

        self.classifier = Classifier(in_shape, n_cls, bn_mode='ds')

        cycle_gan = CycleGAN(hparams={})
        cycle_gan.load_state_dict(torch.load(hparams['cg_ckpt']))

        for params in cycle_gan.parameters():
            params.requires_grad = False
        
        self.cycle_gan = cycle_gan

    def forward(self, x, d):
        y1 = self.classifier(x, d)
        outputs = {
            'x': x,
            'y1_lo': y1
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

        cg_outs = self.model.cycle_gan(x_A=xs, x_B=xt)
        xs = cg_outs['fake_AB']

        outputs = {}
        outputs.update(self.model(xs, 'src'))
        outputs.update(self.model(xt, 'tgt'))

        loss_s = self.loss_fn(outputs['y1_lo_A'], ys)
        loss_t = self.loss_fn(outputs['y1_lo_B'], yt)
        loss = loss_s

        acc_s = (outputs['y1_lo_A'].argmax(-1) == ys).sum() / N
        acc_t = (outputs['y1_lo_B'].argmax(-1) == yt).sum() / N

        loss_dict = {
            'loss': loss,
            'ce_s': loss_s,
            'ce_t': loss_t,
            'acc_s': acc_s,
            'acc_t': acc_t
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

        cg_outs = self.model.cycle_gan(x_A=xs, x_B=xt)
        xs = cg_outs['fake_AB'].detach()

        outputs = {}
        outputs.update(self.model(xs, 'src'))
        outputs.update(self.model(xt, 'tgt'))

        loss_s = self.loss_fn(outputs['y1_lo_A'], ys)
        loss_t = self.loss_fn(outputs['y1_lo_B'], yt)

        loss = loss_s

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        acc_s = (outputs['y1_lo_A'].argmax(-1) == ys).sum() / N
        acc_t = (outputs['y1_lo_B'].argmax(-1) == yt).sum() / N

        loss_dict = {
            'loss': loss,
            'ce_s': loss_s,
            'ce_t': loss_t,
            'acc_s': acc_s,
            'acc_t': acc_t
        }
        return loss_dict
