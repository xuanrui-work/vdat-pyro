from .base_dataset import BaseDataset

import torch
import torchvision
import torchvision.transforms.v2 as v2

class MNIST2USPS(BaseDataset):
    def __init__(
        self,
        reverse=False,
        image_size=(32, 32),
        mnist_save_dir='./cache/mnist', usps_save_dir='./cache/usps',
        **kwargs
    ):
        super().__init__(**kwargs)

        mnist_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(image_size),
            v2.Lambda(self.lambda_repeat),
            v2.ToDtype(torch.float, scale=True),
        ])
        usps_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(image_size),
            v2.Lambda(self.lambda_repeat),
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

        usps_train = torchvision.datasets.USPS(
            root=usps_save_dir,
            train=True,
            transform=usps_transform,
            download=True
        )
        usps_test = torchvision.datasets.USPS(
            root=usps_save_dir,
            train=False,
            transform=usps_transform,
            download=True
        )
        usps_train, usps_val = torch.utils.data.random_split(usps_train, [1-self.val_split, self.val_split])

        if not reverse:
            self.src_train, self.src_val, self.src_test = mnist_train, mnist_val, mnist_test
            self.tgt_train, self.tgt_val, self.tgt_test = usps_train, usps_val, usps_test
        else:
            self.src_train, self.src_val, self.src_test = usps_train, usps_val, usps_test
            self.tgt_train, self.tgt_val, self.tgt_test = mnist_train, mnist_val, mnist_test
    
    # for solving "Can't pickle local object `...<lambda>`" when num_workers > 0 for DataLoader
    def lambda_repeat(self, x):
        return x.repeat(3, 1, 1)
