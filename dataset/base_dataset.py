import torch
from torch.utils.data import DataLoader

import torchvision.transforms.v2 as v2

import numpy as np
import matplotlib.colors as mcolors

from pathlib import Path
from tqdm import tqdm

class BaseDataset:
    def __init__(
        self,
        image_size: tuple[int,int] = (32, 32),
        root: str = None,
        grayscale: bool = False,
        normalize: bool = False,
        batch_size: int = 128,
        val_split: float = 0.2,
        test_split: float = None,
        num_workers: int = 0
    ):
        self.image_size = image_size
        self.root = Path(root)

        self.grayscale = grayscale
        self.normalize = normalize

        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = val_split if test_split is None else test_split

        self.num_workers = num_workers

        self.transforms = []
        if self.grayscale:
            self.transforms += [v2.Lambda(self.lambda_torgb)]

        self.train = None
        self.val = None
        self.test = None

        self.initialized = False
    
    # for solving "Can't pickle local object `...<lambda>`" when num_workers > 0 for DataLoader
    def lambda_torgb(self, x):
        return x.repeat(3, 1, 1)
    
    def loader_factory(self, dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def calc_norm(self):
        loader = self.loader_factory(self.train)
        n = 0
        sum = torch.zeros(3)
        sum_sq = torch.zeros(3)
        for x, _ in tqdm(
            loader,
            desc=f'Computing normalization for {self.__class__.__name__}'
        ):
            n += x.numel() / x.shape[1]
            sum += x.sum((0, 2, 3))
            sum_sq += (x ** 2).sum((0, 2, 3))
        mean = (sum / n).numpy()
        std = torch.sqrt(sum_sq / n - mean ** 2).numpy()
        return (mean, std)
    
    def load_dataset(self, transforms: list[v2.Transform] = None):
        raise NotImplementedError

    def init_dataset(self):
        tfs = [
            v2.ToImage(),
            v2.Resize(self.image_size)
        ] + self.transforms + [v2.ToDtype(torch.float, scale=True)]

        self.load_dataset(tfs)

        if self.normalize:
            try:
                norm_stats = np.load(self.root/'norm_stats.npy')
            except FileNotFoundError:
                norm_stats = self.calc_norm(self.train)
                np.save(self.root/'norm_stats.npy', norm_stats)

            self.load_dataset(self.transforms + [v2.Normalize(*norm_stats)])
            self.norm_stats = norm_stats
        else:
            self.norm_stats = None
        
        self.initialized = True
    
    def get_loaders(self) -> tuple[DataLoader, ...]:
        if not self.initialized:
            self.init_dataset()
        return (
            self.loader_factory(self.train, shuffle=True),
            self.loader_factory(self.val),
            self.loader_factory(self.test)
        )

class VariantDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        self.tfs = kwargs.pop('tfs', None)

        super().__init__(*args, **kwargs)

        if self.tfs is not None:
            self.parse_transforms(self.tfs)
    
    def init_dataset(self):
        tfs = [
            v2.ToImage(),
            v2.Resize(self.image_size)
        ] + self.transforms + [v2.ToDtype(torch.float, scale=True)]

        self.load_dataset(tfs)

        if self.normalize:
            stats_path = (
                self.root/'norm_stats.npy' if self.tfs is None else
                self.root/f"norm_stats--{str(self.tfs).replace(':', '-')}.npy"
            )
            try:
                norm_stats = np.load(self.root/'norm_stats.npy')
            except FileNotFoundError:
                norm_stats = self.calc_norm(self.train)
                np.save(self.root/'norm_stats.npy', norm_stats)

            self.load_dataset(self.transforms + [v2.Normalize(*norm_stats)])
            self.norm_stats = norm_stats
        else:
            self.norm_stats = None
        
        self.initialized = True
    
    def parse_transforms(self, tf_spec: dict):
        if 'colorize' in tf_spec:
            if not self.grayscale:
                raise ValueError('tf_spec: colorize only works with grayscale images')
            
            colors = tf_spec['colorize'] or ['blue', 'yellow']
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
            if len(colors) < 2:
                colors = r_colors
            colors = torch.tensor([mcolors.to_rgb(c) for c in colors])
            colors = (colors * 255).round().to(torch.uint8)

            self.colors = colors
            self.transforms += [v2.Lambda(self.lambda_colorize)]
        
        if 'shrink' in tf_spec:
            self.shrink = tf_spec['shrink'] or 0.5
            self.transforms += [v2.Lambda(self.lambda_shrink)]
        
        if 'rotate' in tf_spec:
            self.rotate_deg = tf_spec['rotate'] or 45
            self.transforms += [v2.Lambda(self.lambda_rotate)]

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
    
    def lambda_shrink(self, x):
        org_size = np.array(self.image_size)
        new_size = (self.shrink * org_size).astype(int)
        x_resized = v2.functional.resize(x, new_size)
        
        x_out = torch.zeros_like(x)
        pad = (org_size - new_size) // 2
        x_out[:, pad[0]:pad[0]+new_size[0], pad[1]:pad[1]+new_size[1]] = x_resized
        return x_out
    
    def lambda_rotate(self, x):
        return v2.functional.rotate(x, self.rotate_deg)

class MultiLoader:
    def __init__(self, datasets: list[BaseDataset]):
        self.datasets = datasets
    
    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return self.datasets[idx]
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.datasets})'
    
    def get_loaders(self) -> tuple[list[iter], ...]:
        train_loaders = []
        val_loaders = []
        test_loaders = []

        for dataset in self.datasets:
            train_loader, val_loader, test_loader = dataset.get_loaders()
            train_loaders += [train_loader]
            val_loaders += [val_loader]
            test_loaders += [test_loader]
        
        return (
            tuple(train_loaders),
            tuple(val_loaders),
            tuple(test_loaders)
        )
