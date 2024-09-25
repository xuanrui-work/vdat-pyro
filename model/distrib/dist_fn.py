import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings

class KLDivDiagonalGaussian(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        p_mu: torch.Tensor, p_var: torch.Tensor,
        q_mu: torch.Tensor, q_var: torch.Tensor
    ):
        p_logvar = torch.log(p_var)
        q_logvar = torch.log(q_var)
        k = p_mu.shape[-1]
        kl_div = 0.5*(
            torch.sum(p_var/q_var, dim=-1) +
            torch.sum((q_mu - p_mu)**2/q_var, dim=-1) +
            torch.sum(q_logvar - p_logvar, dim=-1) - k
        )
        if kl_div.isnan().any():
            warnings.warn('NaN encountered')
        return kl_div

class KLDivFullGaussian(nn.Module):
    def __init__(self, eps=1e-3, nan_to_num=-1):
        super().__init__()
        self.eps = eps
        self.nan_to_num = nan_to_num
    
    def forward(
        self,
        p_mu: torch.Tensor, p_cov: torch.Tensor,
        q_mu: torch.Tensor, q_cov: torch.Tensor
    ):
        # for numerical stability
        p_cov = p_cov + self.eps * torch.eye(p_cov.shape[-1], device=p_cov.device)
        q_cov = q_cov + self.eps * torch.eye(q_cov.shape[-1], device=q_cov.device)

        q_cov_inv = torch.linalg.inv(q_cov)     # (n, k, k)
        diff = (q_mu - p_mu).unsqueeze(-1)      # (n, k, 1)
        diff_T = diff.transpose(-2, -1)         # (n, 1, k)
        k = p_mu.shape[-1]
        kl_div = 0.5*(
            torch.vmap(torch.trace)(q_cov_inv@p_cov) +
            (diff_T@q_cov_inv@diff).squeeze((-1, -2)) +
            torch.logdet(q_cov) - torch.logdet(p_cov) - k
        )
        if self.nan_to_num >= 0:
            kl_div = torch.nan_to_num(kl_div, nan=self.nan_to_num)
        if kl_div.isnan().any():
            warnings.warn('NaN encountered')
        return kl_div

class EntropyDiagonalGaussian(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        mu: torch.Tensor, var: torch.Tensor
    ):
        logvar = torch.log(var)
        k = var.shape[-1]
        h = 0.5*(
            torch.sum(logvar, dim=-1) +
            k*(1 + torch.log(torch.tensor(2*torch.pi, device=var.device)))
        )
        if h.isnan().any():
            warnings.warn('NaN encountered')
        return h

class EntropyFullGaussian(nn.Module):
    def __init__(self, eps=1e-3, nan_to_num=-1):
        super().__init__()
        self.eps = eps
        self.nan_to_num = nan_to_num
    
    def forward(
        self,
        mu: torch.Tensor, cov_L: torch.Tensor
    ):
        # for numerical stability
        cov_L = cov_L + self.eps * torch.eye(cov_L.shape[-1], device=cov_L.device)
        k = cov_L.shape[-1]
        h = 0.5*(
            2*torch.sum(torch.log(torch.diagonal(cov_L, dim1=-2, dim2=-1)), dim=-1) +
            k*(1 + torch.log(torch.tensor(2*torch.pi, device=cov_L.device)))
        )
        if self.nan_to_num >= 0:
            h = torch.nan_to_num(h, nan=self.nan_to_num)
        if h.isnan().any():
            warnings.warn('NaN encountered')
        return h

    def forward1(
        self,
        mu: torch.Tensor, cov: torch.Tensor
    ):
        # for numerical stability
        cov = cov + self.eps * torch.eye(cov.shape[-1], device=cov.device)
        k = cov.shape[-1]
        h = 0.5*(
            torch.logdet(cov) +
            k*(1 + torch.log(torch.tensor(2*torch.pi, device=cov.device)))
        )
        if self.nan_to_num >= 0:
            h = torch.nan_to_num(h, nan=self.nan_to_num)
        if h.isnan().any():
            warnings.warn('NaN encountered')
        return h
