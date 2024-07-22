from .distrib.utils import *
from .network_raw import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro.poutine as poutine

class VDTNet(VDTNetRaw):
    def __init__(self, hparams=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.set_hparams(hparams)
        self.grad_mode('network')
    
    def set_hparams(self, hparams):
        self.hparams = hparams
    
    def requires_grad(self, modules, requires_grad=True):
        if not isinstance(modules, (list, tuple)):
            modules = [modules]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = requires_grad
    
    def grad_mode(self, mode='network'):
        if mode == 'network':
            self.requires_grad([self.enc_zA, self.enc_zB, self.enc_h, self.dec, self.classifier], True)
            self.requires_grad([self.prior_z, self.prior_h], False)
        elif mode == 'priors':
            self.requires_grad([self.enc_zA, self.enc_zB, self.enc_h, self.dec, self.classifier], False)
            self.requires_grad([self.prior_z, self.prior_h], True)
        elif mode == 'classifier':
            self.requires_grad([self.enc_zA, self.enc_zB, self.enc_h, self.dec], False)
            self.requires_grad([self.prior_z, self.prior_h], False)
            self.requires_grad(self.classifier, True)
        else:
            raise ValueError(f'invalid grad mode={mode}')
        self.mode = mode
    
    def forward(self, x, y=None, d='src'):
        # infer latents and outputs
        guide_trace = poutine.trace(self.guide).get_trace(x, y, d)
        model_trace = poutine.trace(poutine.replay(self.model, guide_trace)).get_trace(x, y, d)
        
        z = guide_trace.nodes['z']['value']
        h = guide_trace.nodes['h']['value']
        if y is None:
            y1 = guide_trace.nodes['y']['fn'].probs
            y1_lo = guide_trace.nodes['y']['fn'].logits
        else:
            y1 = F.one_hot(y, self.n_cls).float()
            y1_lo = y1
        x1 = model_trace.nodes['x']['fn'].base_dist.loc

        outputs = {
            'x': x,
            'y': y,
            'z': z,
            'z_mu': guide_trace.nodes['z']['fn'].loc,
            'z_cov_L': guide_trace.nodes['z']['fn'].scale_tril,
            'h': h,
            'h_mu': guide_trace.nodes['h']['fn'].loc,
            'h_cov_L': guide_trace.nodes['h']['fn'].scale_tril,
            'x1': x1,
            'y1': y1,
            'y1_lo': y1_lo
        }

        if d == 'src':
            outputs = {k+'_A': v for k, v in outputs.items()}
        elif d == 'tgt':
            outputs = {k+'_B': v for k, v in outputs.items()}

        # domain translation
        mu1, mu2 = self.prior_z.mu
        cov_L1, cov_L2 = self.prior_z.get_cov_L()

        if d == 'src':
            R_AB, b_AB = find_transform(mu1, cov_L1, mu2, cov_L2)
            z_AB = z @ R_AB.T + b_AB

            guide_trace.nodes['z']['value'] = z_AB
            model_trace = poutine.trace(poutine.replay(self.model, guide_trace)).get_trace(x, y, d='tgt')

            outputs.update({
                'z_AB': z_AB,
                'x_AB': model_trace.nodes['x']['fn'].base_dist.loc
            })
        elif d == 'tgt':
            R_BA, b_BA = find_transform(mu2, cov_L2, mu1, cov_L1)
            z_BA = z @ R_BA.T + b_BA

            guide_trace.nodes['z']['value'] = z_BA
            model_trace = poutine.trace(poutine.replay(self.model, guide_trace)).get_trace(x, y, d='src')

            outputs.update({
                'z_BA': z_BA,
                'x_BA': model_trace.nodes['x']['fn'].base_dist.loc
            })
        
        return outputs
