from torch.utils.data import DataLoader

import torch
import torchvision.transforms.v2 as v2

import numpy as np

from tqdm import tqdm
from pathlib import Path

__all__ = ['BaseDataset']

class BaseDataset:
    def __init__(
        self,
        save_dir_A: str = None,
        save_dir_B: str = None,
        reverse: bool = False,
        normalize: bool = False,
        batch_size: int = 128,
        val_split: float = 0.2,
        test_split: float = None,
        num_workers: int = 0
    ):
        self.save_dir_A = Path(save_dir_A)
        self.save_dir_B = Path(save_dir_B)

        self.reverse = reverse
        self.normalize = normalize
        self.batch_size = batch_size

        self.val_split = val_split
        if test_split is None:
            self.test_split = val_split
        
        self.num_workers = num_workers

        self.train_A = self.val_A = self.test_A = None
        self.train_B = self.val_B = self.test_B = None

        self.transforms_A = []
        self.transforms_B = []

        self.dataset_initialized = False
    
    def load_datasets(self, transforms_A: list, transforms_B: list):
        raise NotImplementedError

    def init_datasets(self):
        self.load_datasets(self.transforms_A, self.transforms_B)

        if self.normalize:
            try:
                norm_A = np.load(self.save_dir_A/'norm.npy')
                norm_B = np.load(self.save_dir_B/'norm.npy')
            except FileNotFoundError:
                norm_A = self.calc_norm(self.train_A)
                norm_B = self.calc_norm(self.train_B)
                np.save(self.save_dir_A/'norm.npy', norm_A)
                np.save(self.save_dir_B/'norm.npy', norm_B)
            
            self.load_datasets(
                self.transforms_A + [v2.Normalize(*norm_A)],
                self.transforms_B + [v2.Normalize(*norm_B)]
            )
            self.norm_A = norm_A
            self.norm_B = norm_B
        else:
            self.norm_A = None
            self.norm_B = None
        
        self.dataset_initialized = True

    def loader_factory(self, dataset, shuffle=False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def calc_norm(self, dataset) -> tuple[np.ndarray, np.ndarray]:
        loader = self.loader_factory(dataset)
        n = 0
        sum = torch.zeros(3)
        sum_sq = torch.zeros(3)
        for x, _ in tqdm(
            loader,
            desc=f'Computing normalization for {dataset.dataset.__class__.__name__}'
        ):
            n += np.prod(x.shape) / x.shape[1]
            sum += x.sum((0, 2, 3))
            sum_sq += (x ** 2).sum((0, 2, 3))
        mean = (sum / n).numpy()
        std = torch.sqrt(sum_sq / n - mean ** 2).numpy()
        return (mean, std)

    def denorm(self, x: torch.Tensor, which: str) -> torch.Tensor:
        assert which in ('src', 'tgt'), f'invalid value for which: {which}'

        if not self.normalize:  # no normalization to undo
            return x

        norms = (
            self.norm_A,
            self.norm_B
        )[::(-1 if self.reverse else 1)]

        norm = norms[int(which == 'tgt')]
        if norm is None:
            raise ValueError('missing normalization statistics. call init_datasets() first')
        
        mean, std = norm
        viewas = (1, -1, 1, 1) if x.ndim == 4 else (-1, 1, 1)

        mean = torch.tensor(mean, device=x.device).view(*viewas)
        std = torch.tensor(std, device=x.device).view(*viewas)
        x = mean + x*std
        return x
    
    def get_loaders(self, which: str) -> tuple[DataLoader, ...]:
        assert which in ('src', 'tgt'), f'invalid value for which: {which}'

        if not self.dataset_initialized:
            self.init_datasets()
        
        datasets = (
            (self.train_A, self.val_A, self.test_A),
            (self.train_B, self.val_B, self.test_B)
        )[::(-1 if self.reverse else 1)]
        
        train, val, test = datasets[int(which == 'tgt')]

        train_loader = self.loader_factory(train, shuffle=True)
        val_loader = self.loader_factory(val)
        test_loader = self.loader_factory(test)

        return (train_loader, val_loader, test_loader)
