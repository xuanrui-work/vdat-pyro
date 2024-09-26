from .bn import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import warnings

class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dims=None,
        activation=lambda: nn.LeakyReLU(0.2, inplace=True),
        batch_norm=False
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

        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class CNN(nn.Module):
    def __init__(
        self,
        in_shape=(3, 32, 32),
        hidden_dims=(64, 128, 256, 512),
        resize=(0, 2, 0, 2),
        activation=lambda: nn.LeakyReLU(0.2, inplace=True),
        batch_norm=False
    ):
        super().__init__()

        in_channels = in_shape[0]
        out_shape = list(in_shape)

        layers = []
        for i, hd in enumerate(hidden_dims):
            layers += [nn.Conv2d(in_channels, hd, kernel_size=3, padding='same')]
            out_shape[0] = hd
            if batch_norm:
                layers += [nn.BatchNorm2d(hd)]
            if resize[i] < 0:
                layers += [nn.MaxPool2d(-resize[i])]
                out_shape[1] //= 2
                out_shape[2] //= 2
            elif resize[i] > 0:
                layers += [nn.Upsample(scale_factor=resize[i])]
                out_shape[1] *= 2
                out_shape[2] *= 2
            layers += [activation()]
            in_channels = hd
        
        self.layers = nn.ModuleList(layers)

        self.in_shape = in_shape
        self.out_shape = tuple(out_shape)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        la_dim=32,
        in_shape=(512, 2, 2),
        out_shape=(3, 32, 32),
        hidden_dims=(512, 256, 128, 64),
        upsample=(2, 2, 2, 2),
        normal_fn=lambda: nn.Sigmoid()
    ):
        super().__init__()

        in_layers = []

        if np.prod(in_shape) != la_dim:
            in_layers += [nn.Linear(la_dim, np.prod(in_shape))]
            warnings.warn(f'incompatible la_dim={la_dim} and in_shape={in_shape}. nn.Linear used to reshape.')
        
        in_layers += [nn.Unflatten(1, in_shape)]
        self.in_layers = nn.Sequential(*in_layers)
        
        cnn_block = CNN(in_shape, hidden_dims, upsample)
        self.cnn_block = cnn_block

        self.bn_layers = nn.ModuleList([DSBatchNorm2d(hd) for hd in hidden_dims])

        out_layers = []

        if cnn_block.out_shape[1:] != tuple(out_shape)[1:]:
            out_layers += [nn.Upsample(size=out_shape[1:])]
            warnings.warn(f'incompatible out_shape={out_shape} and cnn_block.out_shape={cnn_block.out_shape}. nn.Upsample used to reshape.')
        
        out_layers += [nn.Conv2d(hidden_dims[-1], out_shape[0], kernel_size=3, padding='same')]
        out_layers += [normal_fn()]
        self.out_layers = nn.Sequential(*out_layers)

        self.la_dim = la_dim
        self.in_shape = in_shape
        self.out_shape = out_shape
    
    def forward(self, h, d):
        x = self.in_layers(h)
        bn_idx = 0
        for i, layer in enumerate(self.cnn_block.layers):
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                x = self.bn_layers[bn_idx](x, d)
                bn_idx += 1
        x = self.out_layers(x)
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        in_shape=(3, 32, 32),
        la_dim=32,
        hidden_dims=(64, 128, 256, 512),
        maxpools=(2, 2, 2, 2)
    ):
        super().__init__()

        cnn_block = CNN(in_shape, hidden_dims, [-mp for mp in maxpools])
        self.cnn_block = cnn_block

        self.bn_layers = nn.ModuleList([DSBatchNorm2d(hd) for hd in hidden_dims])
        
        in_dim = np.prod(cnn_block.out_shape)
        cov_l_dim = (la_dim * (la_dim + 1)) // 2

        self.op_mu = nn.Linear(in_dim, la_dim)
        self.op_cov_l = nn.Linear(in_dim, cov_l_dim)

        self.in_shape = in_shape
        self.la_dim = la_dim
        self.out_shape = cnn_block.out_shape
    
    def forward(self, x, d):
        bn_idx = 0
        for i, layer in enumerate(self.cnn_block.layers):
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                x = self.bn_layers[bn_idx](x, d)
                bn_idx += 1
        x = x.flatten(1)
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

class Classifier(nn.Module):
    def __init__(
        self,
        in_shape=(3, 32, 32),
        n_cls=10,
        hidden_dims=(64, 128, 256, 512),
        maxpools=(2, 2, 2, 2)
    ):
        super().__init__()

        self.cnn_block = CNN(in_shape, hidden_dims, [-mp for mp in maxpools])
        self.bn_layers = nn.ModuleList([DSBatchNorm2d(hd) for hd in hidden_dims])

        self.op_cls = nn.Linear(np.prod(self.cnn_block.out_shape), n_cls)

        self.in_shape = in_shape
        self.n_cls = n_cls
    
    def foward_repr(self, x, d):
        bn_idx = 0
        for i, layer in enumerate(self.cnn_block.layers):
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                x = self.bn_layers[bn_idx](x, d)
                bn_idx += 1
        return x
    
    def forward(self, x, d):
        x = self.foward_repr(x, d)
        x = x.flatten(1)
        x = self.op_cls(x)
        return x
