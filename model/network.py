import pyro.contrib
from .block import *

from .distrib.priors import *
from .distrib.utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist

import numpy as np

class VDTNetRaw(nn.Module):
    def __init__(
        self,
        in_shape=(3, 32, 32),
        z_dim=32,
        h_dim=32,
        n_cls=10,
        enc_hidden_dims=(512, 256),
        dec_hidden_dims=(256, 256, 512, 512),
        cls_hidden_dims=(512, 256)
    ):
        super().__init__()

        in_dim = np.prod(in_shape)
        la_dim = h_dim + z_dim

        self.in_shape = in_shape
        self.n_cls = n_cls

        self.z_dim = z_dim
        self.h_dim = h_dim

        self.enc_zA = Encoder(in_shape, z_dim, enc_hidden_dims)
        self.enc_zB = Encoder(in_shape, z_dim, enc_hidden_dims)

        self.cls_embed = nn.Sequential(
            MLP(n_cls, np.prod(in_shape[1:])),
            nn.Unflatten(1, (1, *in_shape[1:]))
        )

        in_enc_h = list(in_shape)
        in_enc_h[0] += 1
        self.enc_h = Encoder(in_enc_h, h_dim, enc_hidden_dims)

        self.dec = Decoder(la_dim, in_shape, dec_hidden_dims)

        self.classifier = nn.Sequential(
            nn.Flatten(1),
            MLP(in_dim, n_cls, cls_hidden_dims)
        )

        self.prior_z = CGaussPrior(2, z_dim, requires_grad=True)
        self.prior_h = CGaussPrior(n_cls, h_dim, requires_grad=True)
    
    def model(self, x, y=None, d='src'):
        """
        generative process for the source domain with observed labels.
        """
        # register this module with pyro
        pyro.module('vdt', self)

        N = x.shape[0]
        with pyro.plate('data', N):
            # sample d->z
            if d == 'src':
                mu = self.prior_z.mu[0]
                cov_L = self.prior_z.get_cov_L()[0]
            elif d == 'tgt':
                mu = self.prior_z.mu[1]
                cov_L = self.prior_z.get_cov_L()[1]
            else:
                raise ValueError(f'invalid d={d}')
            z = pyro.sample('z', dist.MultivariateNormal(mu, scale_tril=cov_L))

            # sample y->h
            y = pyro.sample('y', dist.Categorical(logits=self.prior_h.pi).expand((N,)), obs=y)
            mu = self.prior_h.mu[y]
            cov_L = self.prior_h.get_cov_L()[y]
            h = pyro.sample('h', dist.MultivariateNormal(mu, scale_tril=cov_L))

            # sample (z,h)->x
            loc = self.dec(torch.cat([z, h], dim=1))
            # validate_args=False for relaxed Bernoulli values
            pyro.sample('x', dist.Bernoulli(probs=loc, validate_args=False).to_event(3), obs=x)
    
    def guide(self, x, y=None, d='src'):
        N = x.shape[0]
        with pyro.plate('data', N):
            # sample x->z
            if d == 'src':
                z_mu, z_cov_L = self.enc_zA(x)
            elif d == 'tgt':
                z_mu, z_cov_L = self.enc_zB(x)
            else:
                raise ValueError(f'invalid d={d}')
            z = pyro.sample('z', dist.MultivariateNormal(z_mu, scale_tril=z_cov_L))

            # sample x->y
            if y is None:
                logits = self.classifier(x)
                y = pyro.sample('y', dist.Categorical(logits=logits))
            
            # sample (x,y)->h
            y_embed = self.cls_embed(F.one_hot(y, self.n_cls).float())
            h_mu, h_cov_L = self.enc_h(torch.cat([x, y_embed], dim=1))
            h = pyro.sample('h', dist.MultivariateNormal(h_mu, scale_tril=h_cov_L))

    @pyro.contrib.autoname.scope(prefix='cls')
    def model_cls(self, x, y=None):
        """
        auxiliary model for straight-through classification.
        """
        # register this module with pyro
        pyro.module('vdt', self)

        N = x.shape[0]
        with pyro.plate('data', N):
            # sample x->y
            logits = self.classifier(x)
            pyro.sample('y', dist.Categorical(logits=logits), obs=y)
    
    @pyro.contrib.autoname.scope(prefix='cls')
    def guide_cls(self, x, y):
        pass

class VDTNet(VDTNetRaw):
    def __init__(self, hparams=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_hparams(hparams)
    
    def set_hparams(self, hparams):
        self.hparams = hparams
    
    def forward(self, x, y=None, d='src'):
        # infer latents and outputs
        guide_trace = poutine.trace(self.guide).get_trace(x, y, d)
        model_trace = poutine.trace(poutine.replay(self.model, guide_trace)).get_trace(x, y, d)
        
        z = guide_trace.nodes['z']['value']
        h = guide_trace.nodes['h']['value']
        if y is None:
            y1 = guide_trace.nodes['y']['fn'].probs
        else:
            y1 = F.one_hot(y, self.n_cls).float()
        x1 = model_trace.nodes['x']['fn'].base_dist.probs

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
            'y1': y1
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
                'x_AB': model_trace.nodes['x']['fn'].base_dist.probs
            })
        elif d == 'tgt':
            R_BA, b_BA = find_transform(mu2, cov_L2, mu1, cov_L1)
            z_BA = z @ R_BA.T + b_BA

            guide_trace.nodes['z']['value'] = z_BA
            model_trace = poutine.trace(poutine.replay(self.model, guide_trace)).get_trace(x, y, d='src')

            outputs.update({
                'z_BA': z_BA,
                'x_BA': model_trace.nodes['x']['fn'].base_dist.probs
            })
        
        return outputs
