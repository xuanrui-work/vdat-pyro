import torch
from torch.utils.data import DataLoader

__all__ = ['BaseDataset']

class BaseDataset:
    def __init__(
        self,
        batch_size=128,
        val_split=0.2,
        test_split=None,
        num_workers=0
    ):
        if test_split is None:
            test_split = val_split

        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

        self.src_train = self.src_val = self.src_test = None
        self.tgt_train = self.tgt_val = self.tgt_test = None
    
    def get_loaders(self, which):
        assert which in ('src', 'tgt'), (
            f'invalid value for which: {which}'
        )

        train_loader = DataLoader(
            self.src_train if which == 'src' else self.tgt_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            self.src_val if which == 'src' else self.tgt_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            self.src_test if which == 'src' else self.tgt_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return (train_loader, val_loader, test_loader)
