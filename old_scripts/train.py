from model.network import VDTNet
from model.distrib.dist_fn import *

from eval import evaluate
from utils.vis import Visualizer

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.infer
import pyro.poutine

import numpy as np

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset_dict import datasets

from collections import defaultdict

import argparse
import pathlib
import yaml
import warnings

class Trainer:
    def __init__(
        self,
        model_path,
        s_loaders,
        t_loaders,
        device,
        config,
        save_dir='./runs',
        save_stats=True,
        save_vis=True,
        n_vis=50
    ):
        model = VDTNet(in_shape=config['image_size'])
        if model_path:
            model.load_state_dict(torch.load(model_path))
        model.to(device)

        self.model_path = model_path
        self.model = model

        self.s_loaders = s_loaders
        self.t_loaders = t_loaders

        self.device = device
        self.config = config
        
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.save_dir = save_dir

        self.save_stats = save_stats
        self.save_vis = save_vis
        self.n_vis = n_vis

        self.s_train, self.s_val, self.s_test = s_loaders
        self.t_train, self.t_val, self.t_test = t_loaders

        self.writer = SummaryWriter(save_dir)
        self.vis = Visualizer(
            model,
            self.writer,
            device,
            s_loaders[0].batch_size,
            save_dir=save_dir/'vis' if save_vis else ''
        )

        self.stats = {
            'epoch': [],
            'step': [],
            'train': defaultdict(list),
            'train_eval': defaultdict(list),
            'val': defaultdict(list)
        }

        self.vxs_train, self.vys_train = zip(*[self.s_train.dataset[i] for i in range(n_vis)])
        self.vxt_train, self.vyt_train = zip(*[self.t_train.dataset[i] for i in range(n_vis)])

        self.vxs_val, self.vys_val = zip(*[self.s_val.dataset[i] for i in range(n_vis)])
        self.vxt_val, self.vyt_val = zip(*[self.t_val.dataset[i] for i in range(n_vis)])

        self.vxs_test, self.vys_test = zip(*[self.s_test.dataset[i] for i in range(n_vis)])
        self.vxt_test, self.vyt_test = zip(*[self.t_test.dataset[i] for i in range(n_vis)])
    
    def train(self):
        config = self.config
        oparams = self.config['optim']
        hparams = self.config['hparams']

        model = self.model
        device = self.device

        writer = self.writer
        vis = self.vis

        r"""
        Initializations.
        """
        # ELBO loss functions
        elbo_fn_s = pyro.infer.Trace_ELBO().differentiable_loss
        elbo_fn_t = pyro.infer.TraceEnum_ELBO(max_plate_nesting=1).differentiable_loss

        # setup guide for enumeration
        model.enum_guide = pyro.infer.config_enumerate(model.guide, 'sequential', expand=True)

        # setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=oparams['lr'], betas=oparams['betas'])

        # additional loss functions
        entropy_fn = EntropyFullGaussian()

        # max entropy of classifier output
        ent_y_max = torch.distributions.Categorical(
            logits=torch.full((model.n_cls,), 1/model.n_cls, device=device)
        ).entropy()

        num_epochs = config['num_epochs']
        val_every = config['val_every']

        sche_n_steps = oparams['sche_n_steps']
        sche_period = np.sum(sche_n_steps)
        schedule = np.cumsum(sche_n_steps)

        model.set_hparams(hparams)
        model.train()
        step = 0
        best_val_loss = np.inf

        len_loader = min(len(self.s_train), len(self.t_train))

        # print(f'len(s_train_loader): {len(self.s_train)}')
        # print(f'len(t_train_loader): {len(self.t_train)}')
        # print(f'len_loader: {len_loader}')

        r"""
        Training loop.
        """
        for epoch in tqdm(range(num_epochs)):
            for (xs, ys), (xt, yt) in tqdm(
                zip(self.s_train, self.t_train), total=len_loader, leave=False
            ):
            # for (xs, ys), (xt, yt) in zip(self.s_train, self.t_train):
                if xs.shape[0] != xt.shape[0]:
                    continue
                N = xs.shape[0]
                xs, ys = xs.to(device), ys.to(device)
                xt, yt = xt.to(device), yt.to(device)

                sche_step = step % sche_period
                idx = np.sum(sche_step >= schedule)
                if idx == 0:
                    model.grad_mode('network')
                elif idx == 1:
                    model.grad_mode('priors')
                else:
                    raise ValueError(f'index={idx} for schedule is out of range')

                # supervised ELBO loss
                elbo_s = elbo_fn_s(model.model, model.guide, xs, ys, d='src')
                elbo_s /= N
                # unsupervised ELBO loss
                # on src
                elbo_s1 = elbo_fn_t(model.model, model.enum_guide, xs, None, d='src')
                elbo_s1 /= N
                # on tgt
                elbo_t = elbo_fn_t(model.model, model.enum_guide, xt, None, d='tgt')
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
                    nent_y = ent_y / ent_y_max
                    weights = F.softplus(nent_y - hparams['cg_threshold'])

                    l_cls = F.cross_entropy(outputs['y1_lo_A'], ys, reduction='none')
                    l_cls = (weights * l_cls).mean()
                    loss += hparams['w_cg'] * l_cls
                else:
                    l_cls = torch.zeros(1)

                # entropy regularization
                z_all = torch.cat([outputs['z_A'], outputs['z_B']], dim=0)
                ent_z = entropy_fn(z_all.mean(0), torch.cov(z_all.T))
                # loss += hparams['ent_reg'] * -ent_z

                h_all = torch.cat([outputs['h_A'], outputs['h_B']], dim=0)
                ent_h = entropy_fn(h_all.mean(0), torch.cov(h_all.T))
                # loss += hparams['ent_reg'] * -ent_h

                optimizer.zero_grad()
                loss.backward()
                if oparams['clip_grad'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), oparams['clip_grad'])
                optimizer.step()

                loss_dict = {
                    'loss': loss,
                    'elbo_s': elbo_s,
                    'elbo_t': elbo_t,
                    'elbo_s1': elbo_s1,
                    'l_cls_ord': l_cls_ord,
                    'l_cls': l_cls,
                    'ent_z': ent_z,
                    'ent_h': ent_h,
                }

                step += 1
                for k, v in loss_dict.items():
                    writer.add_scalar(f'train/{k}', v.item(), step)
                
                # train statistics
                train_loss_dict = evaluate(model, [(xs, ys)], [(xt, yt)], device, batch_size=N)
                for k, v in train_loss_dict.items():
                    writer.add_scalar(f'train-eval/{k}', v.item(), step)
            
            if epoch % val_every == 0:
                val_loss_dict = evaluate(model, self.s_val, self.t_val, device)
                for k, v in val_loss_dict.items():
                    writer.add_scalar(f'val/{k}', v.item(), step)
                
                # visualize
                vis.vis_samples((self.vxs_train, self.vxt_train), step, 'train', mode='both')
                vis.vis_samples((self.vxs_val, self.vxt_val), step, 'val', mode='both')
                vis.vis_priors(step, 'priors')

                if val_loss_dict['loss'] < best_val_loss:
                    best_val_loss = val_loss_dict['loss']
                    torch.save(model.state_dict(), self.save_dir/'best_model.pth')

                torch.save(model.state_dict(), self.save_dir/'last_model.pth')

                if self.save_stats:
                    self.stats['epoch'] += [epoch]
                    self.stats['step'] += [step]
                    for k, v in loss_dict.items():
                        self.stats['train'][k] += [v.item()]
                    for k, v in train_loss_dict.items():
                        self.stats['train_eval'][k] += [v.item()]
                    for k, v in val_loss_dict.items():
                        self.stats['val'][k] += [v.item()]
                    np.save(self.logdir/'stats.npy', self.stats)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str)

    parser.add_argument('--config', type=str)
    parser.add_argument('--default', type=str, default='default_config.yaml')

    parser.add_argument('--save-dir', type=str, required=True)

    parser.add_argument('--save-stats', action='store_true')
    parser.add_argument('--save-vis', action='store_true')
    parser.add_argument('--n-vis', type=int, default=50)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=False)

    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    with open(args.default, 'r') as f:
        default = yaml.safe_load(f)
    
    for k, v in default.items():
        if k not in config:
            config[k] = v
        else:
            print(f'\t{k}: {default[k]} => {config[k]}')

    dataset = datasets[config['dataset']](
        batch_size=config['batch_size'],
        image_size=config['image_size'][1:],
        val_split=config['val_split']
    )

    r"""supress the warning "classifier.cnn_block.layers.6.bias was not registered in the param store 
    because requires_grad=False." from pyro.
    """
    warnings.filterwarnings(
        'ignore',
        message='.*was not registered in the param store because requires_grad=False',
        module='pyro.primitives'
    )

    Trainer(
        args.model,
        dataset.get_loaders('src'),
        dataset.get_loaders('tgt'),
        device,
        config,
        save_dir,
        args.save_stats,
        args.save_vis,
        args.n_vis
    ).train()

if __name__ == '__main__':
    main()
