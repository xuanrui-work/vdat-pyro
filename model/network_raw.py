from .block_cnn import *
from .distrib.priors import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.contrib
import pyro.distributions as dist

import numpy as np

class VDTNetRaw(nn.Module):
    def __init__(
        self,
        in_shape=(3, 32, 32),
        z_dim=32,
        h_dim=32,
        n_cls=10
    ):
        super().__init__()

        in_dim = np.prod(in_shape)
        la_dim = h_dim + z_dim
        
        self.in_dim = in_dim
        self.la_dim = la_dim

        self.in_shape = in_shape
        self.n_cls = n_cls

        self.z_dim = z_dim
        self.h_dim = h_dim

        self.enc_zA = Encoder(in_shape, z_dim)
        self.enc_zB = Encoder(in_shape, z_dim)

        self.cls_embed = nn.Sequential(
            MLP(n_cls, np.prod(in_shape[1:])),
            nn.Unflatten(1, (1, *in_shape[1:]))
        )

        in_enc_h = list(in_shape)
        in_enc_h[0] += 1
        self.enc_h = Encoder(in_enc_h, h_dim)

        self.dec = Decoder(la_dim, self.enc_h.out_shape, in_shape)

        self.classifier = Classifier(in_shape, n_cls)

        self.prior_z = CGaussPrior(2, z_dim, requires_grad=True)
        self.prior_h = CGaussPrior(n_cls, h_dim, requires_grad=True)

        self.validate_args = True

        # def init_weights(m):
        #     if type(m) == nn.Linear:
        #         nn.init.normal_(m.weight, std=0.01)
        #         nn.init.zeros_(m.bias)
        #     elif type(m) == nn.Conv2d:
        #         nn.init.normal_(m.weight, std=0.01)
        #         nn.init.zeros_(m.bias)
        
        # self.apply(init_weights)
    
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
            z = pyro.sample('z', dist.MultivariateNormal(mu, scale_tril=cov_L).expand((N,)))

            # sample y->h
            y = pyro.sample('y', dist.Categorical(logits=self.prior_h.pi).expand((N,)), obs=y)
            mu = self.prior_h.mu[y]
            cov_L = self.prior_h.get_cov_L()[y]
            h = pyro.sample('h', dist.MultivariateNormal(mu, scale_tril=cov_L))

            # sample (z,h)->x
            loc = self.dec(torch.cat([z, h], dim=1), d)
            pyro.sample(
                'x',
                dist.Normal(
                    loc=loc,
                    scale=1,
                    validate_args=self.validate_args
                ).to_event(3),
                obs=x
            )
    
    def guide(self, x, y=None, d='src'):
        N = x.shape[0]
        with pyro.plate('data', N):
            # sample x->z
            if d == 'src':
                z_mu, z_cov_L = self.enc_zA(x, d)
            elif d == 'tgt':
                z_mu, z_cov_L = self.enc_zB(x, d)
            else:
                raise ValueError(f'invalid d={d}')
            z = pyro.sample(
                'z',
                dist.MultivariateNormal(
                    z_mu,
                    scale_tril=z_cov_L,
                    validate_args=self.validate_args
                )
            )

            # sample x->y
            if y is None:
                logits = self.classifier(x, d)
                y = pyro.sample('y', dist.Categorical(logits=logits))
            
            # sample (x,y)->h
            y_embed = self.cls_embed(F.one_hot(y, self.n_cls).float())
            h_mu, h_cov_L = self.enc_h(torch.cat([x, y_embed], dim=1), d)
            h = pyro.sample(
                'h',
                dist.MultivariateNormal(
                    h_mu,
                    scale_tril=h_cov_L,
                    validate_args=self.validate_args
                )
            )

    @pyro.contrib.autoname.scope(prefix='cls')
    def model_cls(self, x, y=None, d='src'):
        """
        auxiliary model for straight-through classification.
        """
        # register this module with pyro
        pyro.module('vdt', self)

        N = x.shape[0]
        with pyro.plate('data', N):
            # sample x->y
            logits = self.classifier(x, d)
            pyro.sample('y', dist.Categorical(logits=logits), obs=y)
    
    @pyro.contrib.autoname.scope(prefix='cls')
    def guide_cls(self, x, y):
        pass
