from dataset import *

datasets = {
    'mnist2rmnist': lambda **kwargs: MNIST2RMNIST(reverse=False, **kwargs),
    'rmnist2mnist': lambda **kwargs: MNIST2RMNIST(reverse=True, **kwargs),
    'mnist2svhn': lambda **kwargs: MNIST2SVHN(reverse=False, **kwargs),
    'svhn2mnist': lambda **kwargs: MNIST2SVHN(reverse=True, **kwargs),
    'mnist2usps': lambda **kwargs: MNIST2USPS(reverse=False, **kwargs),
    'usps2mnist': lambda **kwargs: MNIST2USPS(reverse=True, **kwargs),
}
