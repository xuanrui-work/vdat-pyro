from .base_dataset import BaseDataset

from .mnist2rmnist import MNIST2RMNIST
from .mnist2svhn import MNIST2SVHN
from .mnist2usps import MNIST2USPS

__all__ = [
    'BaseDataset',
    'MNIST2RMNIST',
    'MNIST2SVHN',
    'MNIST2USPS',
    'dataset_dict'
]

dataset_dict = {
    'mnist2rmnist': lambda **kwargs: MNIST2RMNIST(reverse=False, **kwargs),
    'rmnist2mnist': lambda **kwargs: MNIST2RMNIST(reverse=True, **kwargs),
    'mnist2svhn': lambda **kwargs: MNIST2SVHN(reverse=False, **kwargs),
    'svhn2mnist': lambda **kwargs: MNIST2SVHN(reverse=True, **kwargs),
    'mnist2usps': lambda **kwargs: MNIST2USPS(reverse=False, **kwargs),
    'usps2mnist': lambda **kwargs: MNIST2USPS(reverse=True, **kwargs),
}

def get_dataset_cls(name: str):
    try:
        dataset_cls = dataset_dict[name]
    except KeyError as err:
        raise ValueError(f'invalid dataset name={name}') from err
