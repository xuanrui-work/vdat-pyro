from .base_dataset import BaseDataset

import torch
import torchvision
import torchvision.transforms.v2 as v2

class MNIST2SVHN(BaseDataset):
    def __init__(
        self,
        reverse=False,
        image_size=(32, 32),
        mnist_save_dir='./cache/mnist', svhn_save_dir='./cache/svhn',
        **kwargs
    ):
        super().__init__(**kwargs)

        mnist_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(image_size),
            v2.Lambda(self.lambda_repeat),
            v2.ToDtype(torch.float, scale=True),
        ])
        svhn_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(image_size),
            v2.ToDtype(torch.float, scale=True),
        ])

        mnist_train = torchvision.datasets.MNIST(
            root=mnist_save_dir,
            train=True,
            transform=mnist_transform,
            download=True
        )
        mnist_test = torchvision.datasets.MNIST(
            root=mnist_save_dir,
            train=False,
            transform=mnist_transform,
            download=True
        )
        mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [1-self.val_split, self.val_split])

        svhn_train = torchvision.datasets.SVHN(
            root=svhn_save_dir,
            split='extra',
            transform=svhn_transform,
            download=True
        )
        svhn_train, svhn_test = torch.utils.data.random_split(svhn_train, [1-self.val_split, self.val_split])
        svhn_train, svhn_val = torch.utils.data.random_split(svhn_train, [1-self.val_split, self.val_split])

        if not reverse:
            self.src_train, self.src_val, self.src_test = mnist_train, mnist_val, mnist_test
            self.tgt_train, self.tgt_val, self.tgt_test = svhn_train, svhn_val, svhn_test
        else:
            self.src_train, self.src_val, self.src_test = svhn_train, svhn_val, svhn_test
            self.tgt_train, self.tgt_val, self.tgt_test = mnist_train, mnist_val, mnist_test
    
    # for solving "Can't pickle local object `...<lambda>`" when num_workers > 0 for DataLoader
    def lambda_repeat(self, x):
        return x.repeat(3, 1, 1)
