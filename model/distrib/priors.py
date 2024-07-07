import torch
import torch.nn as nn

class GaussPrior(nn.Module):
    def __init__(
        self,
        dim,
        init='',
        requires_grad=True
    ):
        super().__init__()

        cov_l_dim = dim * (dim + 1) // 2

        self.dim = dim
        self.requires_grad = requires_grad

        if init == '':
            mu_t = torch.zeros(dim)
            cov_l_t = torch.zeros(cov_l_dim)
        elif init == 'randn':
            mu_t = torch.randn(dim)
            cov_l_t = torch.randn(cov_l_dim)
        else:
            raise ValueError(f'invalid init={init}')

        self.mu = nn.Parameter(mu_t, requires_grad=requires_grad)
        self.cov_l = nn.Parameter(cov_l_t, requires_grad=requires_grad)
    
    def get_cov_L(self):
        L = torch.zeros((self.dim, self.dim), device=self.cov_l.device)
        idx = torch.tril_indices(self.dim, self.dim)
        L[idx[0], idx[1]] = self.cov_l
        # ensure positive diagonal
        L.diagonal(dim1=-2, dim2=-1).exp_()
        return L
    
    def get_cov(self):
        L = self.get_cov_L()
        return (L @ L.T)

class CGaussPrior(nn.Module):
    def __init__(
        self,
        nc,
        dim,
        init='',
        requires_grad=True
    ):
        super().__init__()

        cov_l_dim = dim * (dim + 1) // 2

        self.nc = nc
        self.dim = dim
        self.requires_grad = requires_grad

        pi_t = torch.zeros(nc)

        if init == '':
            mu_t = torch.zeros(nc, dim)
            cov_l_t = torch.zeros(nc, cov_l_dim)
        elif init == 'randn':
            mu_t = torch.randn(nc, dim)
            cov_l_t = torch.randn(nc, cov_l_dim)
        else:
            raise ValueError(f'invalid init={init}')

        self.pi = nn.Parameter(pi_t, requires_grad=requires_grad)
        self.mu = nn.Parameter(mu_t, requires_grad=requires_grad)
        self.cov_l = nn.Parameter(cov_l_t, requires_grad=requires_grad)
    
    def get_cov_L(self):
        L = torch.zeros((self.nc, self.dim, self.dim), device=self.cov_l.device)
        idx = torch.tril_indices(self.dim, self.dim)
        L[:, idx[0], idx[1]] = self.cov_l
        # ensure positive diagonal
        L.diagonal(dim1=-2, dim2=-1).exp_()
        return L

    def get_cov(self):
        L = self.get_cov_L()
        return (L @ L.transpose(-2, -1))
