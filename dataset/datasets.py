from .base_dataset import BaseDataset

import torch
import torch.utils.data as data

import torchvision.datasets as tvds
import torchvision.transforms.v2 as v2

import numpy as np
import matplotlib.colors as mcolors

class MNIST(BaseDataset):
    def __init__(
        self,
        image_size: tuple[int,int] = (32, 32),
        root: str = './cache/mnist',
        **kwargs
    ):
        super().__init__(root, **kwargs)

        self.image_size = image_size

        self.transforms = [
            v2.ToImage(),
            v2.Resize(image_size),
            v2.Lambda(self.lambda_repeat),
            v2.ToDtype(torch.float, scale=True),
        ]
    
    # for solving "Can't pickle local object `...<lambda>`" when num_workers > 0 for DataLoader
    def lambda_repeat(self, x):
        return x.repeat(3, 1, 1)
    
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

class CMNIST(MNIST):
    def __init__(
        self,
        image_size: tuple[int,int] = (32, 32),
        colors: tuple[str,...] = ('blue', 'yellow'),
        root: str = './cache/mnist',
        **kwargs
    ):
        super().__init__(image_size, root, **kwargs)

        r_colors = [
            'blue',
            'orange',
            'green',
            'red',
            'purple',
            'brown',
            'pink',
            'gray',
            'olive',
            'cyan',
        ]
        if colors is None or len(colors) < 2:
            colors = r_colors
        colors = torch.tensor([mcolors.to_rgb(c) for c in colors])
        colors = (colors * 255).round().to(torch.uint8)

        self.colors = colors

        self.transforms = [
            v2.ToImage(),
            v2.Resize(image_size),
            v2.Lambda(self.lambda_colorize),
            v2.ToDtype(torch.float, scale=True),
        ]

    # for solving "Can't pickle local object `...<lambda>`" when num_workers > 0 for DataLoader
    def lambda_colorize(self, x):
        Nc = self.colors.shape[0]
        if Nc != 2:
            idx = np.random.choice(Nc, 2, replace=False)
            c_fore, c_back = self.colors[idx]
        else:
            c_fore, c_back = self.colors
        m_fore = x[0] >= 128
        m_back = ~m_fore
        x_out = m_fore*c_fore.view(3, 1, 1) + m_back*c_back.view(3, 1, 1)
        return x_out
    
    def load_dataset(self, transforms):
        super().load_dataset(transforms)

        self.mnist_root = self.root
        self.root = self.root/'rmnist'
        self.root.mkdir(exist_ok=True, parents=True)

class RMNIST(MNIST):
    def __init__(
        self,
        image_size: tuple[int,int] = (32, 32),
        rotate_deg: float = 45,
        root: str = './cache/mnist',
        **kwargs
    ):
        super().__init__(image_size, root, **kwargs)

        self.rotate_deg = rotate_deg

        self.transforms = [
            v2.ToImage(),
            v2.Resize(image_size),
            v2.Lambda(self.lambda_rotate),
            v2.Lambda(self.lambda_repeat),
            v2.ToDtype(torch.float, scale=True),
        ]
    
    # for solving "Can't pickle local object `...<lambda>`" when num_workers > 0 for DataLoader
    def lambda_repeat(self, x):
        return x.repeat(3, 1, 1)
    
    def lambda_rotate(self, x):
        return v2.functional.rotate(x, self.rotate_deg)

    def load_dataset(self, transforms):
        super().load_dataset(transforms)

        self.mnist_root = self.root
        self.root = self.root/'rmnist'
        self.root.mkdir(exist_ok=True, parents=True)

class SVHN(BaseDataset):
    def __init__(
        self,
        image_size: tuple[int,int] = (32, 32),
        root: str = './cache/svhn',
        **kwargs
    ):
        super().__init__(root, **kwargs)

        self.image_size = image_size

        self.transforms = [
            v2.ToImage(),
            v2.Resize(image_size),
            v2.ToDtype(torch.float, scale=True),
        ]
    
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

class USPS(BaseDataset):
    def __init__(
        self,
        image_size: tuple[int,int] = (32, 32),
        root: str = './cache/usps',
        **kwargs
    ):
        super().__init__(root, **kwargs)

        self.image_size = image_size

        self.transforms = [
            v2.ToImage(),
            v2.Resize(image_size),
            v2.Lambda(self.lambda_repeat),
            v2.ToDtype(torch.float, scale=True),
        ]
    
    # for solving "Can't pickle local object `...<lambda>`" when num_workers > 0 for DataLoader
    def lambda_repeat(self, x):
        return x.repeat(3, 1, 1)

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
