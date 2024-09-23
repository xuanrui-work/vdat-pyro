from . import no_adapt

from model.block_cnn import *

from runner.runner import TrainRunner, EvalRunner
from runner import exception
from runner.options import RunnerOptions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(no_adapt.Model):
    pass

class Evaluator(no_adapt.Evaluator):
    pass

class Trainer(no_adapt.Trainer):
    def step(self, xs, ys, xt, yt):
        N = xs.shape[0]

        outputs = {}
        outputs.update(self.model(xs, 'src'))
        outputs.update(self.model(xt, 'tgt'))

        loss_s = self.loss_fn(outputs['y1_lo_A'], ys)
        loss_t = self.loss_fn(outputs['y1_lo_B'], yt)
        loss = (loss_s + loss_t) / 2

        if not isinstance(self, EvalRunner):   # to make compatible with default EvalRunner
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        acc_s = (outputs['y1_lo_A'].argmax(-1) == ys).sum() / N
        acc_t = (outputs['y1_lo_B'].argmax(-1) == yt).sum() / N

        loss_dict = {
            'loss': loss,
            'ce_s': loss_s,
            'ce_t': loss_t,
            'acc': acc_s,
            'acc_t': acc_t
        }
        return loss_dict
