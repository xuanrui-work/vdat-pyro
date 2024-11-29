import torch

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

def find_transform(mu1, cov1, mu2, cov2):
    # torch.linalg.eigh returns sorted (ascending order) eigenvalues and eigenvectors
    L1, Q1 = torch.linalg.eigh(cov1)
    L2, Q2 = torch.linalg.eigh(cov2)
    L12 = torch.diag_embed(torch.sqrt(L2 / L1))
    R12 = Q2 @ L12 @ Q1.T
    b12 = mu2 - R12 @ mu1
    return (R12, b12)

def plot_2d_gaussian(
    mu,
    cov,
    label='',
    color=None,
    level=1,
    res=1000,
    num_samples=1000,
    vis_samples=True,
    **kwargs
):
    mu = np.array(mu)
    cov = np.array(cov)

    assert mu.shape[0] == 2 and cov.shape == (2, 2), (
        'mu must be a 2d vector and cov must be a 2x2 matrix'
    )

    X = np.random.multivariate_normal(mu, cov, num_samples)
    x, y = np.mgrid[
        X[:, 0].min():X[:, 0].max():res*1j,
        X[:, 1].min():X[:, 1].max():res*1j
    ]
    xy = np.dstack((x, y))

    rv = multivariate_normal(mean=mu, cov=cov)
    p = rv.pdf(xy)
    lvl = rv.pdf(mu + np.linalg.cholesky(cov) @ np.array([level, level]))

    if vis_samples:
        plt.scatter(X[:, 0], X[:, 1], alpha=0.5, c=color, **kwargs)
    plt.contour(x, y, p, levels=[lvl], colors=color, **kwargs)
    plt.text(mu[0], mu[1], label, color=color, **kwargs)
