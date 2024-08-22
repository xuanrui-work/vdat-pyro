from .base_dataset import BaseDataset

import torch
import torchvision
import torchvision.transforms.v2 as v2

class MNIST2RMNIST(BaseDataset):
    def __init__(
        self,
        reverse=False,
        rotate_deg=45,
        image_size=(32, 32),
        mnist_save_dir='./cache/mnist',
        **kwargs
    ):
        super().__init__(**kwargs)

        self.rotate_deg = rotate_deg

        mnist_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(image_size),
            v2.Lambda(self.lambda_repeat),
            v2.ToDtype(torch.float, scale=True),
        ])

        rmnist_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(image_size),
            v2.Lambda(self.lambda_rotate),
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

        rmnist_train = torchvision.datasets.MNIST(
            root=mnist_save_dir,
            train=True,
            transform=rmnist_transform,
            download=True
        )
        rmnist_test = torchvision.datasets.MNIST(
            root=mnist_save_dir,
            train=False,
            transform=rmnist_transform,
            download=True
        )
        rmnist_train, rmnist_val = torch.utils.data.random_split(rmnist_train, [1-self.val_split, self.val_split])

        if not reverse:
            self.src_train, self.src_val, self.src_test = mnist_train, mnist_val, mnist_test
            self.tgt_train, self.tgt_val, self.tgt_test = rmnist_train, rmnist_val, rmnist_test
        else:
            self.src_train, self.src_val, self.src_test = rmnist_train, rmnist_val, rmnist_test
            self.tgt_train, self.tgt_val, self.tgt_test = mnist_train, mnist_val, mnist_test
    
    # for solving "Can't pickle local object `...<lambda>`" when num_workers > 0 for DataLoader
    def lambda_repeat(self, x):
        return x.repeat(3, 1, 1)
    
    def lambda_rotate(self, x):
        return v2.functional.rotate(x, self.rotate_deg)
