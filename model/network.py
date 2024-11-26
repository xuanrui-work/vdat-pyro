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
        self.trans_mode('sample')
    
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
    
    def trans_mode(self, mode='sample'):
        if mode not in ('sample', 'mean'):
            raise ValueError(f'invalid inference mode={mode}')
        self.mode_trans = mode
    
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
        tod = 'tgt'
        tag = 'AB'

        if d == 'tgt':  # swap the means and covariances for translating from target to source
            t1, t2 = mu2, cov_L2
            mu2, cov_L2 = mu1, cov_L1
            mu1, cov_L1 = t1, t2
            tod = 'src'
            tag = 'BA'

        R_AB, b_AB = find_transform(mu1, cov_L1, mu2, cov_L2)

        if self.mode_trans == 'mean':
            z = guide_trace.nodes['z']['fn'].loc
            h = guide_trace.nodes['h']['fn'].loc
        z_AB = z @ R_AB.T + b_AB
        h_AB = h

        guide_trace.nodes['z']['value'] = z_AB
        guide_trace.nodes['h']['value'] = h_AB
        model_trace = poutine.trace(poutine.replay(self.model, guide_trace)).get_trace(x, y, d=tod)

        outputs.update({
            'z_'+tag: z_AB,
            'x_'+tag: model_trace.nodes['x']['fn'].base_dist.loc
        })
        
        return outputs
