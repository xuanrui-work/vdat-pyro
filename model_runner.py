from model.distrib.dist_fn import EntropyFullGaussian

from runner.runner import EvalRunner, TrainRunner
from utils.vis import Visualizer

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.infer
import pyro.poutine

import numpy as np

import warnings

class Evaluator(EvalRunner):
    def __init__(
        self,
        model,
        save_dir='',
        progbar=False,
        options=None
    ):
        super().__init__(
            model,
            save_dir,
            progbar,
            options
        )
    
    def before_run(self):
        super().before_run()
        self.elbo_fn = pyro.infer.Trace_ELBO().differentiable_loss
    
    def step(self, xs, ys, xt, yt):
        model = self.model
        N = xs.shape[0]

        # evaluate ELBO loss
        elbo_s = self.elbo_fn(model.model, model.guide, xs, ys, 'src')
        elbo_t = self.elbo_fn(model.model, model.guide, xt, yt, 'tgt')
        loss = (elbo_s + elbo_t) / 2

        # evaluate classifier accuracy
        model_trace = pyro.poutine.trace(model.model_cls).get_trace(xs, d='src')
        ys_pred = model_trace.nodes['cls/y']['fn'].logits

        model_trace = pyro.poutine.trace(model.model_cls).get_trace(xt, d='tgt')
        yt_pred = model_trace.nodes['cls/y']['fn'].logits

        acc_s = (ys_pred.argmax(-1) == ys).sum()
        acc_t = (yt_pred.argmax(-1) == yt).sum()

        loss_dict = {
            'loss': loss,
            'elbo_s': elbo_s,
            'elbo_t': elbo_t,
            'acc_s': acc_s,
            'acc_t': acc_t
        }
        for k, v in loss_dict.items():
            loss_dict[k] = v / N
        return loss_dict

class Trainer(TrainRunner):
    def __init__(
        self,
        model,
        save_dir,
        progbar=False,
        options=None
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
    
    def init_visualize(self):
        self.vis = Visualizer(
            self.model,
            self.writer,
            self.device,
            self.options.batch_size,
            save_dir=self.save_dir/'vis'
        )

        get_vis = lambda loader: list(zip(*[loader.dataset[i] for i in range(self.options.n_vis)]))

        self.vis_data = {
            'train': [get_vis(loader) for loader in self.train_loaders],
            'val': [get_vis(loader) for loader in self.val_loaders],
            'test': [get_vis(loader) for loader in self.test_loaders]
        }
    
    def visualize(self):
        vis_train = self.vis_data['train']
        vis_val = self.vis_data['val']

        self.vis.vis_samples((vis_train[0][0], vis_train[1][0]), self.n_step, 'train', mode='both')
        self.vis.vis_samples((vis_val[0][0], vis_val[1][0]), self.n_step, 'val', mode='both')
        self.vis.vis_priors(self.n_step, 'priors')
    
    def before_run(self):
        super().before_run()

        self.all_traine_stats = {}

        r"""supress the warning "... was not registered in the param store 
        because requires_grad=False" from pyro.
        """
        warnings.filterwarnings(
            'ignore',
            message='.*was not registered in the param store because requires_grad=False',
            module='pyro.primitives'
        )

        hparams = self.options.hparams
        oparams = self.options.oparams
        model = self.model

        self.init_visualize()

        # ELBO loss functions
        self.elbo_fn_s = pyro.infer.Trace_ELBO().differentiable_loss
        self.elbo_fn_t = pyro.infer.TraceEnum_ELBO(max_plate_nesting=1).differentiable_loss

        # setup guide for enumeration
        model.enum_guide = pyro.infer.config_enumerate(model.guide, 'sequential', expand=True)

        # setup optimizer
        self.optim = torch.optim.Adam(
            model.parameters(),
            lr=oparams.lr,
            betas=oparams.betas
        )

        # additional loss functions
        self.entropy_fn = EntropyFullGaussian()

        # pre-defined default entropy for h and z
        self.ent_h_def = self.entropy_fn(torch.zeros(model.h_dim), torch.eye(model.h_dim))
        self.ent_z_def = self.entropy_fn(torch.zeros(model.z_dim), torch.eye(model.z_dim))

        # max entropy of classifier output
        self.ent_y_max = torch.distributions.Categorical(
            logits=torch.full((model.n_cls,), 1/model.n_cls, device=self.device)
        ).entropy()

        # schedule parameters
        self.sche_n_steps = self.options.oparams.sche_n_steps
        self.sche_period = np.sum(self.sche_n_steps)
        self.schedule = np.cumsum(self.sche_n_steps)

        # initialize model
        model.set_hparams(hparams)

        # initial visualization
        self.visualize()
    
    def step(self, xs, ys, xt, yt):
        self.last_batch = (xs, ys, xt, yt)

        # print(xs.shape, ys.shape)
        # print(xt.shape, yt.shape)

        N = xs.shape[0]
        model = self.model
        hparams = self.options.hparams
        oparams = self.options.oparams

        # determine gradient mode
        sche_step = self.n_step % self.sche_period
        idx = np.sum(sche_step >= self.schedule)
        if idx == 0:
            model.grad_mode('network')
        elif idx == 1:
            model.grad_mode('priors')
        else:
            raise ValueError(f'index={idx} for schedule is out of range')
        
        # supervised ELBO loss
        elbo_s = self.elbo_fn_s(model.model, model.guide, xs, ys, d='src')
        elbo_s /= N
        # unsupervised ELBO loss
        # on src
        elbo_s1 = self.elbo_fn_t(model.model, model.enum_guide, xs, None, d='src')
        elbo_s1 /= N
        # on tgt
        elbo_t = self.elbo_fn_t(model.model, model.enum_guide, xt, None, d='tgt')
        elbo_t /= N
        # total loss
        loss = (
            hparams['w_domain'][0]/2 * (elbo_s + elbo_s1) +
            hparams['w_domain'][1] * elbo_t
        )

        # additional forward pass
        outputs = {}
        outputs.update(model(xs, None, d='src'))
        outputs.update(model(xt, None, d='tgt'))

        # ordinary classifier loss
        l_cls_ord = F.cross_entropy(outputs['y1_lo_A'], ys)
        
        # direct classifier guidance
        if hparams['w_cg'] > 0:
            ent_y = torch.distributions.Categorical(logits=outputs['y1_lo_A'].detach()).entropy()
            nent_y = ent_y / self.ent_y_max
            weights = F.softplus(nent_y - hparams['cg_threshold'])

            l_cls = F.cross_entropy(outputs['y1_lo_A'], ys, reduction='none')
            l_cls = (weights * l_cls).mean()
            loss += hparams['w_cg'] * l_cls
        else:
            l_cls = torch.zeros(1)

        # entropy regularization
        ent_zA = self.entropy_fn(None, outputs['z_cov_L_A'])
        ent_zB = self.entropy_fn(None, outputs['z_cov_L_B'])

        ent_hA = self.entropy_fn(None, outputs['h_cov_L_A'])
        ent_hB = self.entropy_fn(None, outputs['h_cov_L_B'])

        if hparams['w_er'] > 0:
            er = (
                F.relu(hparams['er_ratio'] - ent_zA.min()/self.ent_z_def) +
                F.relu(hparams['er_ratio'] - ent_zB.min()/self.ent_z_def) +
                F.relu(hparams['er_ratio'] - ent_hA.min()/self.ent_h_def) +
                F.relu(hparams['er_ratio'] - ent_hB.min()/self.ent_h_def)
            )**2
            loss += hparams['w_er'] * er
        
        # label-domain relative entropy regularization
        # TODO: DOES NOT WORK
        if hparams['w_rer'] > 0:
            avgent_h = (ent_hA.mean() + ent_hB.mean()) / 2
            avgent_z = (ent_zA.mean() + ent_zB.mean()) / 2
            rer = (avgent_h/avgent_z - hparams['rer_ratio'])**2
            loss += hparams['w_rer'] * rer
        
        # cross-domain relative entropy regularization
        if hparams['w_rer_d'] > 0:
            rer_d = (ent_zB.mean() / ent_zA.mean() - hparams['rer_d_ratio'])**2
            loss += hparams['w_rer_d'] * rer_d

        self.optim.zero_grad()
        loss.backward()
        if oparams['clip_grad'] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), oparams['clip_grad'])
        self.optim.step()

        loss_dict = {
            'loss': loss,
            'elbo_s': elbo_s,
            'elbo_t': elbo_t,
            'elbo_s1': elbo_s1,
            'l_cls_ord': l_cls_ord,
            'l_cls': l_cls,
            'ent_zA': ent_zA.mean(),
            'ent_zB': ent_zB.mean(),
            'ent_hA': ent_hA.mean(),
            'ent_hB': ent_hB.mean()
        }
        return loss_dict

    def after_step(self):
        super().after_step()

        # more train stats
        xs, ys, xt, yt = self.last_batch
        
        traine_stats = self.eval_runner.execute([[(xs, ys)], [(xt, yt)]])
        self.traine_stats = traine_stats.copy()

        # logging
        for k, v in traine_stats.items():
            self.writer.add_scalar(f'train-eval/{k}', v, self.n_step)
        
        # update stats
        traine_stats['step'] = self.n_step
        for k, v in traine_stats.items():
            if k not in self.all_traine_stats:
                self.all_traine_stats[k] = []
            self.all_traine_stats[k] += [v]
    
    def after_run(self):
        super().after_run()

        # visualize test samples
        vis_test = self.vis_data['test']
        self.vis.vis_samples((vis_test[0][0], vis_test[1][0]), self.n_step, 'val', mode='both')

        # save train-eval stats
        np.save(self.save_dir/'traineval_stats.npy', self.all_traine_stats)
