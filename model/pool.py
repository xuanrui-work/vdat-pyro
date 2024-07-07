"""
Implementation of a buffer for storing past latent vectors and labels, which becomes useful when 
we update the prior distribution of the latent space.
"""

import torch

class TensorPool:
    def __init__(
        self,
        keys=('l', 'c'),
        dims=(32, 10),
        max_len=5000,
        device='cpu'
    ):
        self.keys = keys
        self.dims = dims

        self.max_len = max_len
        self.device = device

        self.pool = {k: [] for k in keys}
    
    def add(self, *args, **kwargs):
        if args:
            kwargs1 = dict(zip(self.keys, args))
            kwargs1.update(kwargs)
            kwargs = kwargs1
        for k, v in kwargs.items():
            N = v.shape[0]
            while len(self.pool[k]) + N > self.max_len:
                self.pool[k].pop(0)
            self.pool[k] += [v.detach().to(self.device)]
    
    def clr(self):
        self.pool = {k: [] for k in self.keys}
    
    def cat(self, *args):
        if not args:
            args = self.keys
        outs = []
        for k in args:
            assert len(self.pool[k]) > 0, f'empty pool["{k}"], nothing to concatenate'
            outs += [
                torch.cat(self.pool[k], dim=0)
            ]
        return tuple(outs)
