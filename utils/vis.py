from model.distrib.utils import *

import torchvision.utils as tvutils
import pathlib

class Visualizer:
    def __init__(
        self,
        model,
        writer,
        device,
        batch_size=64,
        image_keys=('x_A', 'x_B', 'x1_A', 'x1_B', 'x_AB', 'x_BA'),
        embed_keys=(('z_A', 'z_B', 'z_AB', 'z_BA'), ('h_A', 'h_B')),
        save_dir=''
    ):
        self.model = model
        self.writer = writer

        self.device = device
        self.batch_size = batch_size
        
        self.image_keys = image_keys
        self.embed_keys = embed_keys

        if save_dir:
            save_dir = pathlib.Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = None
        self.save_dir = save_dir
    
    def vis_samples(self, samples, step, tag, labels=None, mode='both'):
        outputs_all = {}

        training = self.model.training
        self.model.eval()

        # collect outputs
        with torch.no_grad():
            for i in range(0, len(samples), self.batch_size):
                if mode == 'src' or mode == 'tgt':
                    x = torch.stack(samples[i:i+self.batch_size]).to(self.device)
                    if labels is not None:
                        y = torch.stack(labels[i:i+self.batch_size]).to(self.device)
                    else:
                        y = None
                    outputs = self.model(x, y, d=mode)
                
                elif mode == 'both':
                    x_A = torch.stack(samples[0][i:i+self.batch_size]).to(self.device)
                    x_B = torch.stack(samples[1][i:i+self.batch_size]).to(self.device)
                    if labels is not None:
                        y_A = torch.stack(labels[0][i:i+self.batch_size]).to(self.device)
                        y_B = torch.stack(labels[1][i:i+self.batch_size]).to(self.device)
                    else:
                        y_A = None
                        y_B = None
                    outputs = {}
                    outputs.update(self.model(x_A, y_A, d='src'))
                    outputs.update(self.model(x_B, y_B, d='tgt'))
                
                else:
                    raise ValueError(f'invalid mode={mode}')
                
                for k, v in outputs.items():
                    if v is None:
                        continue
                    if k not in outputs_all:
                        outputs_all[k] = []
                    outputs_all[k] += [v]
        
        self.model.train(training)
        
        # concatenate output tensors
        for k, v in outputs_all.items():
            outputs_all[k] = torch.cat(v, dim=0)
        
        # visualize images
        for k in self.image_keys:
            if k in outputs_all:
                self.writer.add_images(f'{tag}/{k}', outputs_all[k], step)
                if self.save_dir:
                    self.save_images(outputs_all[k], step, f'{tag}/{k}')
        
        # visualize embeddings
        for group in self.embed_keys:
            g_embed_mat = []
            g_metadata = []
            g_label_img = []
            g_tags = []

            for k in group:
                if k in outputs_all:
                    embed = outputs_all[k].flatten(1)
                    g_embed_mat += [embed]
                    g_metadata += [k] * embed.shape[0]
                    g_tags += [k]
                    if k.endswith('AB'):
                        g_label_img += [outputs_all['x_AB']]
                    elif k.endswith('BA'):
                        g_label_img += [outputs_all['x_BA']]
                    elif k.endswith('A'):
                        g_label_img += [outputs_all['x_A']]
                    elif k.endswith('B'):
                        g_label_img += [outputs_all['x_B']]
                    else:
                        g_label_img += [torch.zeros(embed.shape[0], *samples[0].shape)]
            
            g_embed_mat = torch.cat(g_embed_mat, dim=0)
            g_label_img = torch.cat(g_label_img, dim=0)

            self.writer.add_embedding(
                g_embed_mat,
                metadata=g_metadata,
                label_img=g_label_img,
                global_step=step,
                tag=f'{tag}/{",".join(g_tags)}'
            )
    
    def vis_priors(self, step, tag, dims=(0, 1)):
        dims = list(dims)

        mu_zA, mu_zB = self.model.prior_z.mu.numpy(force=True)
        cov_zA, cov_zB = self.model.prior_z.get_cov().numpy(force=True)

        mu_h = self.model.prior_h.mu.numpy(force=True)[0]
        cov_h = self.model.prior_h.get_cov().numpy(force=True)[0]

        fig = plt.figure(figsize=(8, 8))
        plot_2d_gaussian(mu_zA[dims], cov_zA[dims, :][:, dims], label='prior_zA')
        plot_2d_gaussian(mu_zB[dims], cov_zB[dims, :][:, dims], label='prior_zB')
        plot_2d_gaussian(mu_h[dims], cov_h[dims, :][:, dims], label='prior_h')
        self.writer.add_figure(tag, fig, step)
        # fig.clear()
        plt.close(fig)

    def save_images(self, images, step, tag):
        grid_img = tvutils.make_grid(images, nrow=10)
        tag = tag.replace('/', '_')
        filepath = self.save_dir / f'{tag}_{step}.png'
        tvutils.save_image(grid_img, filepath)
