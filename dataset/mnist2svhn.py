from .base_dataset import BaseDataset

import torch
import torch.utils.data as data

import torchvision.datasets as datasets
import torchvision.transforms.v2 as v2

__all__ = [
    'MNIST2SVHN'
]

class MNIST2SVHN(BaseDataset):
    def __init__(
        self,
        image_size: tuple[int,int] = (32, 32),
        save_dir_A: str = './cache/mnist',
        save_dir_B: str = './cache/svhn',
        **kwargs
    ):
        super().__init__(
            save_dir_A=save_dir_A,
            save_dir_B=save_dir_B,
            **kwargs
        )

        self.image_size = image_size

        self.transforms_A = [
            v2.ToImage(),
            v2.Resize(self.image_size),
            v2.Lambda(self.lambda_repeat),
            v2.ToDtype(torch.float, scale=True),
        ]

        self.transforms_B = [
            v2.ToImage(),
            v2.Resize(self.image_size),
            v2.ToDtype(torch.float, scale=True),
        ]
    
    # for solving "Can't pickle local object `...<lambda>`" when num_workers > 0 for DataLoader
    def lambda_repeat(self, x):
        return x.repeat(3, 1, 1)
    
    def init_mnist(self, transforms):
        train = datasets.MNIST(
            root=self.save_dir_A,
            train=True,
            transform=v2.Compose(transforms),
            download=True
        )
        test = datasets.MNIST(
            root=self.save_dir_A,
            train=False,
            transform=v2.Compose(transforms),
            download=True
        )
        train, val = data.random_split(train, [1-self.val_split, self.val_split])

        self.train_A = train
        self.val_A = val
        self.test_A = test
    
    def init_svhn(self, transforms):
        train = datasets.SVHN(
            root=self.save_dir_B,
            split='extra',
            transform=v2.Compose(transforms),
            download=True
        )
        train, test = data.random_split(train, [1-self.test_split, self.test_split])
        train, val = data.random_split(train, [1-self.val_split, self.val_split])

        self.train_B = train
        self.val_B = val
        self.test_B = test

    def load_datasets(self, transforms_A, transforms_B):
        self.init_mnist(transforms_A)
        self.init_svhn(transforms_B)
