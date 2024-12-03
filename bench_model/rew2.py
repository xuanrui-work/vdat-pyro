from model.block_cnn import *
from .rew1 import Model as DomainClassifier

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

        self.d_cls = DomainClassifier()
        self.d_cls.load_state_dict(torch.load(hparams['dcls_ckpt']))

        # freeze domain classifier
        for param in self.d_cls.parameters():
            param.requires_grad = False

    def forward(self, x, d):
        y1 = self.classifier(x, d)
        d1 = self.d_cls.classifier(x, d)
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

        hparams = self.options.hparams

        outputs = {}
        outputs.update(self.model(xs, 'src'))
        outputs.update(self.model(xt, 'tgt'))

        loss_s = self.loss_fn(outputs['y1_lo_A'], ys)
        loss_t = self.loss_fn(outputs['y1_lo_B'], yt)

        pd_pred = F.softmax(outputs['d1_lo_A'].detach(), dim=-1)
        ps = pd_pred[:,0]
        pt = pd_pred[:,1]

        beta = torch.clip(pt / ps, min=hparams['beta_r'][0], max=hparams['beta_r'][1])
        mean_log_beta = torch.mean(torch.log(beta))

        loss = beta * F.cross_entropy(outputs['y1_lo_A'], ys, reduction='none')
        loss = torch.mean(loss)

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
            'acc_t': acc_t,
            'mean_log_beta': mean_log_beta
        }
        return loss_dict
