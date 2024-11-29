from runner.runner import TrainRunner, EvalRunner
from runner import exception
from runner.options import RunnerOptions

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (3, 32, 32),
        hidden_dims: list[int] = None,
        max_pools: list[int] = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]

        self.input_shape = input_shape
        self.hidden_dims = hidden_dims
        self.max_pools = max_pools
        
        conv_layers = []
        output_shape = list(input_shape)

        for i, hidden_dim in enumerate(hidden_dims):
            conv_layers += [nn.Conv2d(
                output_shape[0],
                hidden_dim,
                kernel_size=3,
                padding='same'
            )]
            if max_pools[i]:
                conv_layers += [nn.MaxPool2d(kernel_size=max_pools[i])]
                output_shape[1] //= max_pools[i]
                output_shape[2] //= max_pools[i]

            conv_layers += [nn.BatchNorm2d(hidden_dim)]
            conv_layers += [nn.LeakyReLU(0.2, inplace=True)]
            output_shape[0] = hidden_dim
        
        output_shape = tuple(output_shape)

        self.conv_layers = nn.Sequential(*conv_layers)
        self.output_shape = output_shape

    def forward(self, x):
        x = self.conv_layers(x)
        return x

class CNNDecoder(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        hidden_dims: list[int] = None,
        upsamples: list[int] = None,
        output_shape: tuple[int, int, int] = (3, 32, 32),
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512][::-1]

        self.input_shape = input_shape

        conv_layers = []
        output_shape1 = list(input_shape)

        for i, hidden_dim in enumerate(hidden_dims):
            if upsamples[i]:
                conv_layers += [nn.Upsample(scale_factor=upsamples[i], mode='bilinear')]
                output_shape1[1] *= upsamples[i]
                output_shape1[2] *= upsamples[i]
            conv_layers += [nn.Conv2d(
                output_shape1[0],
                hidden_dim,
                kernel_size=3,
                padding='same'
            )]

            conv_layers += [nn.BatchNorm2d(hidden_dim)]
            conv_layers += [nn.LeakyReLU(0.2, inplace=True)]
            output_shape1[0] = hidden_dim
        
        output_shape1 = tuple(output_shape1)
        
        self.conv_layers = nn.Sequential(*conv_layers)

        output_layer = [
            nn.Conv2d(hidden_dims[-1], output_shape[0], kernel_size=3, padding='same'),
            nn.Sigmoid()
        ]
        if output_shape1[1:] != output_shape[1:]:
            output_layer.insert(
                0,
                nn.Upsample(size=output_shape[1:], mode='bilinear')
            )
            warnings.warn(
                f'Mismatch between model output shape output_shape1={output_shape1} and output_shape={output_shape}. '
                f'A nn.Upsample has been prepended to the output layer to give the desired output_shape.'
            )
        
        self.output_layer = nn.Sequential(*output_layer)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.output_layer(x)
        return x

class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        input_shape=(3, 32, 32),
        hidden_dims=(64, 128, 256),
        max_pools=(2, 2, 2),
        hparams=None
    ):
        super().__init__()
        self.hparams = hparams

        layers = []
        output_shape = list(input_shape)

        encoder = CNNEncoder(
            input_shape,
            hidden_dims,
            max_pools
        )
        layers += [encoder]
        output_shape = list(encoder.output_shape)
        
        layers += [nn.Conv2d(output_shape[0], 1, kernel_size=3, padding='same')]
        output_shape[0] = 1

        self.layers = nn.Sequential(*layers)
        self.output_shape = tuple(output_shape)
    
    def forward(self, x):
        return self.layers(x)

class Generator(nn.Module):
    def __init__(
        self,
        input_shape=(3, 32, 32),
        hidden_dims=(64, 128, 128),
        max_pools=(2, 2, 0),
        hparams=None
    ):
        super().__init__()
        self.hparams = hparams

        self.encoder = CNNEncoder(
            input_shape,
            hidden_dims,
            max_pools
        )
        self.decoder = CNNDecoder(
            self.encoder.output_shape,
            hidden_dims[::-1],
            max_pools[::-1],
            input_shape
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class CycleGAN(nn.Module):
    def __init__(
        self,
        hparams=None
    ):
        super().__init__()

        self.G_AB = Generator(hparams=hparams)
        self.G_BA = Generator(hparams=hparams)
        self.D_A = PatchDiscriminator(hparams=hparams)
        self.D_B = PatchDiscriminator(hparams=hparams)

        self.set_hparams(hparams)

        # loss functions
        self.criter_gan = GANLoss('lsgan')
        self.criter_cyc = nn.L1Loss()

        # loss dict
        self.loss_dict = {}
        self.init_optimizers()

    def init_optimizers(self):
        hparams = self.hparams

        # optimizers
        self.optim_G = torch.optim.Adam(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            lr=hparams['lr'],
            betas=(hparams['betas'][0], hparams['betas'][1])
        )
        self.optim_D = torch.optim.Adam(
            list(self.D_A.parameters()) + list(self.D_B.parameters()),
            lr=hparams['lr'],
            betas=(hparams['betas'][0], hparams['betas'][1])
        )
    
    def set_hparams(self, hparams):
        if 'cyc' in hparams:
            hparams['cyc_ABA'] = hparams['cyc']
            hparams['cyc_BAB'] = hparams['cyc']
        self.hparams = hparams

        self.G_AB.hparams = hparams
        self.G_BA.hparams = hparams
        self.D_A.hparams = hparams
        self.D_B.hparams = hparams
    
    def set_requires_grad(self, nets, requires_grad=True):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad
    
    def forward(self, x_A=None, x_B=None):
        assert x_A is not None or x_B is not None, (
            f'either x_A or x_B must be provided, but got x_A={x_A} and x_B={x_B}'
        )

        if x_A is not None:
            fake_AB = self.G_AB(x_A)
            fake_ABA = self.G_BA(fake_AB)
        else:
            fake_AB = fake_ABA = None
        if x_B is not None:
            fake_BA = self.G_BA(x_B)
            fake_BAB = self.G_AB(fake_BA)
        else:
            fake_BA = fake_BAB = None

        outputs = {
            'real_A': x_A,
            'real_B': x_B,
            'fake_AB': fake_AB,
            'fake_BA': fake_BA,
            'fake_ABA': fake_ABA,
            'fake_BAB': fake_BAB
        }
        return outputs

    def backward_D(self, D, real, fake):
        # real
        p_real = D(real)
        l_real = self.criter_gan(p_real, True)
        # fake
        p_fake = D(fake.detach())
        l_fake = self.criter_gan(p_fake, False)
        # total loss
        loss_D = 0.5 * (l_real + l_fake)
        return loss_D

    def backward_D_A(self, outputs):
        fake_A = outputs['fake_BA']
        l_D_A = self.backward_D(self.D_A, outputs['real_A'], fake_A)
        self.loss_dict['l_D_A'] = l_D_A
        return l_D_A

    def backward_D_B(self, outputs):
        fake_B = outputs['fake_AB']
        l_D_B = self.backward_D(self.D_B, outputs['real_B'], fake_B)
        self.loss_dict['l_D_B'] = l_D_B
        return l_D_B

    def backward_G(self, outputs):
        # GAN loss D_B(G_AB(A))
        l_G_AB = self.criter_gan(self.D_B(outputs['fake_AB']), True)
        # GAN loss D_A(G_BA(B))
        l_G_BA = self.criter_gan(self.D_A(outputs['fake_BA']), True)
        # || G_BA(G_AB(A)) - A ||
        l_cyc_ABA = self.criter_cyc(outputs['fake_ABA'], outputs['real_A'])
        # || G_AB(G_BA(B)) - B ||
        l_cyc_BAB = self.criter_cyc(outputs['fake_BAB'], outputs['real_B'])
        # total loss
        loss_G = (
            l_G_AB +
            l_G_BA +
            self.hparams['cyc_ABA'] * l_cyc_ABA +
            self.hparams['cyc_BAB'] * l_cyc_BAB
        )

        self.loss_dict.update({
            'l_G_AB': l_G_AB,
            'l_G_BA': l_G_BA,
            'l_cyc_A': l_cyc_ABA,
            'l_cyc_B': l_cyc_BAB
        })
        return loss_G
    
    def optimize_params(self, x_A, x_B, backward=True):
        # forward pass
        outputs = self.forward(x_A, x_B)

        # update D_A and D_B
        self.set_requires_grad([self.D_A, self.D_B], True)
        l_D_A = self.backward_D_A(outputs)
        l_D_B = self.backward_D_B(outputs)
        if backward:
            self.optim_D.zero_grad()
            l_D_A.backward()
            l_D_B.backward()
            self.optim_D.step()

        # update G_A and G_B
        self.set_requires_grad([self.D_A, self.D_B], False)
        loss_G = self.backward_G(outputs)
        if backward:
            self.optim_G.zero_grad()
            loss_G.backward()
            self.optim_G.step()

        return (outputs, self.loss_dict)

class Visualizer:
    def __init__(self, model, writer, device, batch_size=64):
        self.model = model
        self.writer = writer
        self.device = device
        self.batch_size = batch_size
    
    def vis_samples(self, samples, step, tag, mode='ab'):
        outputs_all = {}

        training = self.model.training
        self.model.eval()

        with torch.no_grad():
            for i in range(0, len(samples), self.batch_size):
                x = torch.stack(samples[i:i+self.batch_size]).to(self.device)
                if mode == 'ab':
                    outputs = self.model(x_A=x, x_B=None)
                elif mode == 'ba':
                    outputs = self.model(x_A=None, x_B=x)
                else:
                    raise ValueError(f'invalid mode={mode}')
                for k, v in outputs.items():
                    if v is None:
                        continue
                    if k not in outputs_all:
                        outputs_all[k] = []
                    outputs_all[k] += [v]
        
        self.model.train(training)
        
        for k, v in outputs_all.items():
            outputs_all[k] = torch.cat(v, dim=0)
            self.writer.add_images(f'{tag}/{k}', outputs_all[k], step)

Model = CycleGAN

class Evaluator(EvalRunner):
    def step(self, xs, ys, xt, yt):
        outputs, loss_dict = self.model.optimize_params(xs, xt, backward=False)
        return loss_dict

class Trainer(TrainRunner):
    def __init__(
        self,
        model: CycleGAN,
        save_dir: str,
        progbar: bool = False,
        options: RunnerOptions|dict = None
    ):
        super().__init__(
            model,
            save_dir,
            Evaluator(
                model,
                '',
                progbar,
                options
            ),
            progbar,
            options
        )
    
    def init_visualize(self):
        self.vis = Visualizer(
            self.model,
            self.writer,
            self.device,
            self.options.batch_size,
            save_dir=self.save_dir/'vis'
        )

        get_vis = lambda loader: list(zip(*[loader.dataset[i] for i in range(self.options.n_vis)]))

        self.vis_data = {
            'train': [get_vis(loader) for loader in self.train_loaders],
            'val': [get_vis(loader) for loader in self.val_loaders],
            'test': [get_vis(loader) for loader in self.test_loaders]
        }
    
    def visualize(self):
        vis_train = self.vis_data['train']
        vis_val = self.vis_data['val']

        self.vis.vis_samples(vis_train[0][0], self.n_step, 'train', mode='ab')
        self.vis.vis_samples(vis_train[1][0], self.n_step, 'train', mode='ba')

        self.vis.vis_samples(vis_val[0][0], self.n_step, 'val', mode='ab')
        self.vis.vis_samples(vis_val[1][0], self.n_step, 'val', mode='ba')

    def before_run(self):
        super().before_run()

        # setup optimizer
        self.model.init_optimizers()

    def step(self, xs, ys, xt, yt):
        outputs, loss_dict = self.model.optimize_params(xs, xt)
        return loss_dict
