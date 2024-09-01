import torch
from torch.utils.data import DataLoader

import torchvision.transforms.v2 as v2

import numpy as np

from pathlib import Path
from tqdm import tqdm

class BaseDataset:
    def __init__(
        self,
        root: str = None,
        transforms: list[v2.Transform] = None,
        normalize: bool = False,
        batch_size: int = 128,
        val_split: float = 0.2,
        test_split: float = None,
        num_workers: int = 0
    ):
        self.root = Path(root)
        self.transforms = transforms

        self.normalize = normalize
        self.batch_size = batch_size

        self.val_split = val_split
        self.test_split = val_split if test_split is None else test_split

        self.num_workers = num_workers

        self.train = None
        self.val = None
        self.test = None

        self.initialized = False
    
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
        self.load_dataset(self.transforms)

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
