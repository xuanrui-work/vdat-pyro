from .base_dataset import BaseDataset

from .mnist2rmnist import MNIST2RMNIST
from .mnist2svhn import MNIST2SVHN
from .mnist2usps import MNIST2USPS

__all__ = [
    'BaseDataset',
    'MNIST2RMNIST',
    'MNIST2SVHN',
    'MNIST2USPS'
]
