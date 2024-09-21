from .base_dataset import VariantDataset

import torch
import torch.utils.data as data

import torchvision.datasets as tvds
import torchvision.transforms.v2 as v2

class MNIST(VariantDataset):
    def __init__(
        self,
        image_size: tuple[int,int] = (32, 32),
        root: str = './cache/mnist',
        *args,
        **kwargs
    ):
        super().__init__(image_size, root, True, *args, **kwargs)
    
    def load_dataset(self, transforms):
        train = tvds.MNIST(
            root=self.root,
            train=True,
            transform=v2.Compose(transforms),
            download=True
        )
        test = tvds.MNIST(
            root=self.root,
            train=False,
            transform=v2.Compose(transforms),
            download=True
        )
        train, val = data.random_split(train, [1-self.val_split, self.val_split])

        self.train = train
        self.val = val
        self.test = test

class SVHN(VariantDataset):
    def __init__(
        self,
        image_size: tuple[int,int] = (32, 32),
        root: str = './cache/svhn',
        *args,
        **kwargs
    ):
        super().__init__(image_size, root, *args, **kwargs)
    
    def load_dataset(self, transforms):
        train = tvds.SVHN(
            root=self.root,
            split='extra',
            transform=v2.Compose(transforms),
            download=True
        )
        train, test = data.random_split(train, [1-self.test_split, self.test_split])
        train, val = data.random_split(train, [1-self.val_split, self.val_split])

        self.train = train
        self.val = val
        self.test = test

class USPS(VariantDataset):
    def __init__(
        self,
        image_size: tuple[int,int] = (32, 32),
        root: str = './cache/usps',
        *args,
        **kwargs
    ):
        super().__init__(image_size, root, True, *args, **kwargs)

    def load_dataset(self, transforms):
        train = tvds.USPS(
            root=self.root,
            train=True,
            transform=v2.Compose(transforms),
            download=True
        )
        test = tvds.USPS(
            root=self.root,
            train=False,
            transform=v2.Compose(transforms),
            download=True
        )
        train, val = data.random_split(train, [1-self.val_split, self.val_split])

        self.train = train
        self.val = val
        self.test = test

class OfficeHome(VariantDataset):
    def __init__(
        self,
        image_size: tuple[int,int] = (32, 32),
        root: str = './cache/officehome',
        which: str = 'Real World',
        *args,
        **kwargs
    ):
        super().__init__(image_size, root, *args, **kwargs)
        self.which = which
    
    def load_dataset(self, transforms):
        train = tvds.ImageFolder(
            root=f'{self.root}/{self.which}',
            transform=v2.Compose(transforms)
        )
        train, test = data.random_split(train, [1-self.test_split, self.test_split])
        train, val = data.random_split(train, [1-self.val_split, self.val_split])

        self.train = train
        self.val = val
        self.test = test
