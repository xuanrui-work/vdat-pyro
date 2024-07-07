from .block import *

from .distrib.priors import *
from .distrib.utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist

import numpy as np

class VDTNetRaw(nn.Module):
    def __init__(
        self,
        in_shape=(3, 32, 32),
        z_dim=32,
        h_dim=32,
        n_cls=10,
        hidden_dims=(512, 256)
    ):
        super().__init__()

        in_dim = np.prod(in_shape)
        la_dim = h_dim + z_dim

        self.in_shape = in_shape
        self.n_cls = n_cls

        self.z_dim = z_dim
        self.h_dim = h_dim

        self.enc_zA = Encoder(in_shape, z_dim, hidden_dims)
        self.enc_zB = Encoder(in_shape, z_dim, hidden_dims)
        self.enc_h = Encoder(in_shape, h_dim, hidden_dims)

        self.dec = Decoder(la_dim, in_shape, hidden_dims[::-1])
        self.classifier = MLP(h_dim, n_cls, hidden_dims)

        self.prior_z = CGaussPrior(2, z_dim)
        self.prior_h = CGaussPrior(n_cls, h_dim)
    
    def model(self, x, d, y=None):
        # register this module with pyro
        pyro.module('vdt', self)

        N = x.shape[0]
        with pyro.plate('data', N):
            # sample d->z
            d = pyro.sample('d', dist.Categorical(logits=self.prior_z.pi).expand((N,)), obs=d)
            mu = self.prior_z.mu[d]
            cov_L = self.prior_z.get_cov_L()[d]
            z = pyro.sample('z', dist.MultivariateNormal(mu, scale_tril=cov_L))
            # sample y->h
            y = pyro.sample('y', dist.Categorical(logits=self.prior_h.pi).expand((N,)), obs=y)
            mu = self.prior_h.mu[y]
            cov_L = self.prior_h.get_cov_L()[y]
            h = pyro.sample('h', dist.MultivariateNormal(mu, scale_tril=cov_L))
            # sample (z,h)->x
            la = torch.cat([z, h], dim=-1)
            loc = self.dec(la)
            # validate_args=False for relaxed Bernoulli values
            pyro.sample('x', dist.Bernoulli(probs=loc, validate_args=False).to_event(3), obs=x)
        # return model outputs
        return loc
    
    def guide(self, x, d, y=None):
        N = x.shape[0]
        with pyro.plate('data', N):
            # sample x->z
            zA_mu, zA_cov_L = self.enc_zA(x)
            zB_mu, zB_cov_L = self.enc_zB(x)
            z_mu = (
                (d == 0).float().view(N, 1) * zA_mu +
                (d == 1).float().view(N, 1) * zB_mu
            )
            z_cov_L = (
                (d == 0).float().view(N, 1, 1) * zA_cov_L +
                (d == 1).float().view(N, 1, 1) * zB_cov_L
            )
            z = pyro.sample('z', dist.MultivariateNormal(z_mu, scale_tril=z_cov_L))
            # sample x->h
            h_mu, h_cov_L = self.enc_h(x)
            h = pyro.sample('h', dist.MultivariateNormal(h_mu, scale_tril=h_cov_L))
            # # sample h->y if y is not observed
            if y is None:
                logits = self.classifier(h)
                pyro.sample('y', dist.Categorical(logits=logits))
    
    def model_classify(self, x, y):
        # register this module with pyro
        pyro.module('vdt', self)

        N = x.shape[0]
        with pyro.plate('data_cls', N):
            # sample h->y
            mu = self.prior_h.mu[y]
            cov_L = self.prior_h.get_cov_L()[y]
            h = pyro.sample('h_cls', dist.MultivariateNormal(mu, scale_tril=cov_L))
            # sample h->y
            logits = self.classifier(h)
            with pyro.poutine.scale(scale=self.hparams['cls']):
                pyro.sample('y_cls', dist.Categorical(logits=logits), obs=y)
        # return model outputs
        return logits
    
    def guide_classify(self, x, y):
        N = x.shape[0]
        with pyro.plate('data_cls', N):
            # sample x->h
            h_mu, h_cov_L = self.enc_h(x)
            h = pyro.sample('h_cls', dist.MultivariateNormal(h_mu, scale_tril=h_cov_L))

class VDTNet(VDTNetRaw):
    def forward(self, x_A=None, x_B=None):
        assert x_A is not None or x_B is not None, (
            f'either x_A or x_B must be provided, but got x_A={x_A} and x_B={x_B}'
        )

        training = self.training
        self.eval()

        if x_A is not None:
            with torch.no_grad():
                # infer z and h
                trace = pyro.poutine.trace(self.guide).get_trace(x_A, x_A.new_zeros(x_A.shape[0]))
                z_A = trace.nodes['z']['value']
                h_A = trace.nodes['h']['value']
                # reconstruct x
                x_A1 = self.dec(torch.cat([z_A, h_A], dim=-1))
                # classify: infer y from h
                logits_A = self.classifier(h_A)
                y_A = F.softmax(logits_A, dim=-1)

                # translate domain
                R_AB, b_AB = find_transform(
                    self.prior_z.mu[0], self.prior_z.get_cov()[0],
                    self.prior_z.mu[1], self.prior_z.get_cov()[1]
                )
                z_AB = z_A @ R_AB.T + b_AB
                x_AB = self.dec(torch.cat([z_AB, h_A], dim=-1))
        else:
            z_A = None
            h_A = None
            y_A = None
            x_A1 = None
            z_AB = None
            x_AB = None
        
        if x_B is not None:
            with torch.no_grad():
                # infer z and h
                trace = pyro.poutine.trace(self.guide).get_trace(x_B, x_B.new_ones(x_B.shape[0]))
                z_B = trace.nodes['z']['value']
                h_B = trace.nodes['h']['value']
                # reconstruct x
                x_B1 = self.dec(torch.cat([z_B, h_B], dim=-1))
                # classify: infer y from h
                logits_B = self.classifier(h_B)
                y_B = F.softmax(logits_B, dim=-1)

                # translate domain
                R_BA, b_BA = find_transform(
                    self.prior_z.mu[1], self.prior_z.get_cov()[1],
                    self.prior_z.mu[0], self.prior_z.get_cov()[0]
                )
                z_BA = z_B @ R_BA.T + b_BA
                x_BA = self.dec(torch.cat([z_BA, h_B], dim=-1))
        else:
            z_B = None
            h_B = None
            y_B = None
            x_B1 = None
            z_BA = None
            x_BA = None
        
        self.train(training)
        outputs = {
            'x_A': x_A,
            'x_B': x_B,
            'z_A': z_A,
            'z_B': z_B,
            'h_A': h_A,
            'h_B': h_B,
            'y_A': y_A,
            'y_B': y_B,
            'x_A1': x_A1,
            'x_B1': x_B1,
            'z_AB': z_AB,
            'z_BA': z_BA,
            'x_AB': x_AB,
            'x_BA': x_BA
        }
        return outputs

    def reparametrize(self, mu, cov_L):
        eps = torch.randn_like(mu)
        z = mu + (eps.unsqueeze(1) @ cov_L.mT).squeeze(1)
        return z
    
    def forward(self, x_A=None, x_B=None):
        assert x_A is not None or x_B is not None, (
            f'either x_A or x_B must be provided, but got x_A={x_A} and x_B={x_B}'
        )

        if x_A is not None:
            z_A_mu, z_A_cov_L = self.enc_zA(x_A)
            h_A_mu, h_A_cov_L = self.enc_h(x_A)
            # individually sample p(z) and p(h) is equivalent to sampling p(z, h)
            # given independence of z and h
            z_A = self.reparametrize(z_A_mu, z_A_cov_L)
            h_A = self.reparametrize(h_A_mu, h_A_cov_L)
            y_A = self.classifier(h_A)
            x_A1 = self.dec(torch.cat([z_A, h_A], dim=1))
            
            R_AB, b_AB = find_transform(
                self.prior_z.mu[0], self.prior_z.get_cov()[0],
                self.prior_z.mu[1], self.prior_z.get_cov()[1]
            )
            z_AB = z_A @ R_AB.T + b_AB
            x_AB = self.dec(torch.cat([z_AB, h_A], dim=1))
        else:
            z_A = z_A_mu = z_A_cov_L = None
            h_A = h_A_mu = h_A_cov_L = None
            z_AB = None
            x_A1 = None
            x_AB = None
        
        if x_B is not None:
            z_B_mu, z_B_cov_L = self.enc_zB(x_B)
            h_B_mu, h_B_cov_L = self.enc_h(x_B)
            z_B = self.reparametrize(z_B_mu, z_B_cov_L)
            h_B = self.reparametrize(h_B_mu, h_B_cov_L)
            y_B = self.classifier(h_B)
            x_B1 = self.dec(torch.cat([z_B, h_B], dim=1))
            
            R_BA, b_BA = find_transform(
                self.prior_z.mu[1], self.prior_z.get_cov()[1],
                self.prior_z.mu[0], self.prior_z.get_cov()[0]
            )
            z_BA = z_B @ R_BA.T + b_BA
            x_BA = self.dec(torch.cat([z_BA, h_B], dim=1))
        else:
            z_B = z_B_mu = z_B_cov_L = None
            h_B = h_B_mu = h_B_cov_L = None
            z_BA = None
            x_B1 = None
            x_BA = None
        
        outputs = {
            'x_A': x_A,
            'x_B': x_B,
            'z_A': z_A, 'z_A_mu': z_A_mu, 'z_A_cov_L': z_A_cov_L,
            'z_B': z_B, 'z_B_mu': z_B_mu, 'z_B_cov_L': z_B_cov_L,
            'h_A': h_A, 'h_A_mu': h_A_mu, 'h_A_cov_L': h_A_cov_L,
            'h_B': h_B, 'h_B_mu': h_B_mu, 'h_B_cov_L': h_B_cov_L,
            'y_A': y_A,
            'y_B': y_B,
            'x_A1': x_A1,
            'x_B1': x_B1,
            'z_AB': z_AB,
            'z_BA': z_BA,
            'x_AB': x_AB,
            'x_BA': x_BA
        }
        return outputs
