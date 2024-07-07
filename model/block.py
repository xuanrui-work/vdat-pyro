import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dims=None,
        activation=lambda: nn.LeakyReLU(0.2, inplace=True),
        batch_norm=True
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = []
        self.in_dim = in_dim
        self.out_dim = out_dim

        in_features = in_dim
        layers = []
        for i, hd in enumerate(hidden_dims):
            layers += [nn.Linear(in_features, hd)]
            if batch_norm:
                layers += [nn.BatchNorm1d(hd)]
            layers += [activation()]
            in_features = hd
        layers += [nn.Linear(in_features, out_dim)]

        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(
        self,
        la_dim=32,
        out_shape=(3, 32, 32),
        hidden_dims=(256, 256, 512, 512),
        normal_fn=lambda: nn.Sigmoid()
    ):
        super().__init__()

        out_dim = np.prod(out_shape)

        self.la_dim = la_dim
        self.out_shape = out_shape
        self.out_dim = out_dim

        self.layers = MLP(la_dim, out_dim, hidden_dims)
        self.normal_fn = normal_fn()
    
    def forward(self, h):
        x = self.layers(h).view(-1, *self.out_shape)
        x = self.normal_fn(x)
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        in_shape=(3, 32, 32),
        la_dim=32,
        hidden_dims=(512, 256)
    ):
        super().__init__()

        in_dim = np.prod(in_shape)
        cov_l_dim = (la_dim * (la_dim + 1)) // 2

        self.in_shape = in_shape
        self.in_dim = in_dim
        self.la_dim = la_dim
        self.cov_l_dim = cov_l_dim

        self.layers = MLP(in_dim, la_dim, hidden_dims)

        self.op_mu = nn.Linear(la_dim, la_dim)
        self.op_cov_l = nn.Linear(la_dim, cov_l_dim)
    
    def forward(self, x):
        x = self.layers(x.flatten(1))
        mu = self.op_mu(x)
        cov_l = self.op_cov_l(x)
        
        # convert cov_l into the cholesky matrix
        dim = self.la_dim
        L = torch.zeros((x.shape[0], dim, dim), device=x.device)
        idx = torch.tril_indices(dim, dim)
        L[:, idx[0], idx[1]] = cov_l
        # ensure positive diagonal
        L.diagonal(dim1=-2, dim2=-1).exp_()

        return (mu, L)
