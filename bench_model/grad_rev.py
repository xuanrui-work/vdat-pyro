from model.block_cnn import *

from runner.runner import TrainRunner, EvalRunner
from runner import exception
from runner.options import RunnerOptions

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class GradientReversalF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class Model(nn.Module):
    def __init__(
        self,
        in_shape=(3, 32, 32),
        n_cls=10,
        hparams=None
    ):
        super().__init__()

        self.in_shape = in_shape
        self.n_cls = n_cls
        self.hparams = hparams

        self.classifier = Classifier(in_shape, n_cls)
        self.domain_cls_head = nn.Linear(np.prod(self.classifier.cnn_block.out_shape), 2)

    def forward(self, x, d):
        x = self.classifier.foward_repr(x, d)
        x = x.flatten(1)
        y1 = self.classifier.op_cls(x)

        x_rev = GradientReversalF.apply(x, self.hparams['alpha'])
        d1 = self.domain_cls_head(x_rev)

        outputs = {
            'x': x,
            'y1_lo': y1,
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

        loss_s = self.loss_fn(outputs['y1_lo_A'], ys)
        loss_t = self.loss_fn(outputs['y1_lo_B'], yt)
        loss = loss_s

        acc_s = (outputs['y1_lo_A'].argmax(-1) == ys).sum() / N
        acc_t = (outputs['y1_lo_B'].argmax(-1) == yt).sum() / N

        acc_ds = (outputs['d1_lo_A'].argmax(-1) == 0).sum() / N
        acc_dt = (outputs['d1_lo_B'].argmax(-1) == 1).sum() / N

        loss_dict = {
            'loss': loss,
            'ce_s': loss_s,
            'ce_t': loss_t,
            'acc_s': acc_s,
            'acc_t': acc_t,
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

        alpha = 2 / (1 + np.exp(-10 * self.n_step/self.num_steps)) - 1
        self.model.hparams['alpha'] = alpha

        outputs = {}
        outputs.update(self.model(xs, 'src'))
        outputs.update(self.model(xt, 'tgt'))

        loss_s = self.loss_fn(outputs['y1_lo_A'], ys)
        loss_t = self.loss_fn(outputs['y1_lo_B'], yt)

        ds = torch.tensor([0]*N, device=xs.device)
        dt = torch.tensor([1]*N, device=xt.device)
        ld_s = self.loss_fn(outputs['d1_lo_A'], ds)
        ld_t = self.loss_fn(outputs['d1_lo_B'], dt)

        loss = loss_s + (ld_s + ld_t) / 2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        acc_s = (outputs['y1_lo_A'].argmax(-1) == ys).sum() / N
        acc_t = (outputs['y1_lo_B'].argmax(-1) == yt).sum() / N

        acc_ds = (outputs['d1_lo_A'].argmax(-1) == 0).sum() / N
        acc_dt = (outputs['d1_lo_B'].argmax(-1) == 1).sum() / N

        loss_dict = {
            'loss': loss,
            'ce_s': loss_s,
            'ce_t': loss_t,
            'acc_s': acc_s,
            'acc_t': acc_t,
            'ld_s': ld_s,
            'ld_t': ld_t,
            'acc_ds': acc_ds,
            'acc_dt': acc_dt
        }
        return loss_dict
