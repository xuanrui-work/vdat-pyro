r"""
WeightedBatchNorm gives different weights to samples from different domains when estimating statistics.

The implementation is derived from BatchNorm2d in PyTorch.
Available at https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d.

Credit to https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py for an example implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings

class DSBatchNorm2d(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True
    ):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats
        )

        self.register_buffer("num_batches_tracked_A", torch.tensor(0, dtype=torch.long))
        self.register_buffer("running_mean_A", torch.zeros(num_features))
        self.register_buffer("running_var_A", torch.ones(num_features))
        
        self.register_buffer("num_batches_tracked_B", torch.tensor(0, dtype=torch.long))
        self.register_buffer("running_mean_B", torch.zeros(num_features))
        self.register_buffer("running_var_B", torch.ones(num_features))
    
    def forward(self, x, d='src'):
        self._check_input_dim(x)

        if d == 'src':
            num_batches_tracked = self.num_batches_tracked_A
            running_mean = self.running_mean_A
            running_var = self.running_var_A
        elif d == 'tgt':
            num_batches_tracked = self.num_batches_tracked_B
            running_mean = self.running_mean_B
            running_var = self.running_var_B
        else:
            raise ValueError(f'invalid d={d}, expected "src" or "tgt"')
        
        exponential_average_factor = 0.0
        
        if self.training and self.track_running_stats:
            if num_batches_tracked is not None:
                num_batches_tracked.add_(1)
                if self.momentum is None:   # use cumulative moving average
                    exponential_average_factor = 1.0 / float(num_batches_tracked)
                else:   # use exponential moving average
                    exponential_average_factor = self.momentum
        
        if self.training or not self.track_running_stats:
            mean = x.mean([0, 2, 3])
            var = x.var([0, 2, 3], unbiased=False)
            n = x.numel() / x.shape[1]
            with torch.no_grad():
                running_mean.copy_(
                    exponential_average_factor * mean +
                    (1 - exponential_average_factor) * running_mean
                )
                running_var.copy_(
                    exponential_average_factor * var * n / (n - 1) +
                    (1 - exponential_average_factor) * running_var
                )

        else:
            mean = running_mean
            var = running_var
        
        x = (x - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        
        return x

r"""
test to ensure that the implementation is correct
borrowed from https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
"""
def compare_bn(bn1, bn2):
    err = False
    if not torch.allclose(bn1.running_mean_A, bn2.running_mean):
        print('Diff in running_mean: {} vs {}'.format(
            bn1.running_mean_A, bn2.running_mean))
        err = True

    if not torch.allclose(bn1.running_var_A, bn2.running_var):
        print('Diff in running_var: {} vs {}'.format(
            bn1.running_var_A, bn2.running_var))
        err = True

    if bn1.affine and bn2.affine:
        if not torch.allclose(bn1.weight, bn2.weight):
            print('Diff in weight: {} vs {}'.format(
                bn1.weight, bn2.weight))
            err = True

        if not torch.allclose(bn1.bias, bn2.bias):
            print('Diff in bias: {} vs {}'.format(
                bn1.bias, bn2.bias))
            err = True

    if not err:
        print('All parameters are equal!')

if __name__ == '__main__':
    # init BatchNorm layers
    my_bn = DSBatchNorm2d(3, affine=True)
    bn = nn.BatchNorm2d(3, affine=True)

    compare_bn(my_bn, bn)  # weight and bias should be different

    # load weight and bias
    with torch.no_grad():
        bn.weight.copy_(my_bn.weight)
        bn.bias.copy_(my_bn.bias)

    compare_bn(my_bn, bn)

    # run train
    for _ in range(100):
        scale = torch.randint(1, 10, (1,)).float()
        bias = torch.randint(-10, 10, (1,)).float()

        r"""
        with Ns=Nt=N/2 and d_weight=(0.5, 0.5), 
        the my_bn and bn should have the same behavior and output
        """
        N = 20
        x = torch.randn(N, 3, 100, 100) * scale + bias
        d = torch.cat((
            torch.tensor(0).expand(N//2),
            torch.tensor(1).expand(N//2)
        ))
        d_weight = (0.5, 0.5)

        out1 = my_bn(x)
        out2 = bn(x)
        compare_bn(my_bn, bn)

        torch.allclose(out1, out2)
        print('Max diff: ', (out1 - out2).abs().max())

    # run eval
    my_bn.eval()
    bn.eval()
    for _ in range(100):
        scale = torch.randint(1, 10, (1,)).float()
        bias = torch.randint(-10, 10, (1,)).float()
        
        N = 20
        x = torch.randn(N, 3, 100, 100) * scale + bias
        d = torch.cat((
            torch.tensor(0).expand(N//2),
            torch.tensor(1).expand(N//2)
        ))
        d_weight = (0.5, 0.5)

        out1 = my_bn(x)
        out2 = bn(x)
        compare_bn(my_bn, bn)

        torch.allclose(out1, out2)
        print('Max diff: ', (out1 - out2).abs().max())
