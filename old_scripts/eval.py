from model.network import VDTNet

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.infer
import pyro.poutine

from tqdm import tqdm

from dataset_dict import datasets

import argparse
import yaml
import pathlib

def evaluate(
    model,
    s_loader,
    t_loader,
    device,
    batch_size=None
):
    elbo_fn = pyro.infer.Trace_ELBO().differentiable_loss
    loss_dict = {}

    len_loader = min(len(s_loader), len(t_loader))
    if batch_size is None:
        batch_size = s_loader.batch_size

    training = model.training
    model.eval()
    
    with torch.no_grad():
        for (xs, ys), (xt, yt) in tqdm(
            zip(s_loader, t_loader), total=len_loader, leave=False
        ):
            if xs.shape[0] != xt.shape[0]:
                continue
            N = xs.shape[0]
            xs, ys = xs.to(device), ys.to(device)
            xt, yt = xt.to(device), yt.to(device)

            # evaluate ELBO loss
            elbo_s = elbo_fn(model.model, model.guide, xs, ys, 'src')
            elbo_t = elbo_fn(model.model, model.guide, xt, yt, 'tgt')
            loss = (elbo_s + elbo_t) / 2

            # evaluate classifier accuracy
            model_trace = pyro.poutine.trace(model.model_cls).get_trace(xs, d='src')
            ys_pred = model_trace.nodes['cls/y']['fn'].logits

            model_trace = pyro.poutine.trace(model.model_cls).get_trace(xt, d='tgt')
            yt_pred = model_trace.nodes['cls/y']['fn'].logits

            acc_s = (ys_pred.argmax(-1) == ys).sum()
            acc_t = (yt_pred.argmax(-1) == yt).sum()

            loss_dict1 = {
                'loss': loss,
                'elbo_s': elbo_s,
                'elbo_t': elbo_t,
                'acc_s': acc_s,
                'acc_t': acc_t
            }
            for k in loss_dict1:
                if k not in loss_dict:
                    loss_dict[k] = 0
                loss_dict[k] = loss_dict[k] + loss_dict1[k]
    
    model.train(training)

    for k in loss_dict:
        loss_dict[k] = loss_dict[k] / (len_loader*batch_size)
    return loss_dict

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=datasets.keys())
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--vis', action='store_true')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=False)

    dataset = datasets[args.dataset](
        batch_size=args.batch_size,
        val_split=0
    )
    s_train, s_val, s_test = dataset.get_loaders('src')
    t_train, t_val, t_test = dataset.get_loaders('tgt')

    model = VDTNet().load_state_dict(torch.load(args.model)).to(device)

    results = {}
    results['train'] = evaluate(model, s_train, t_train, device)
    # results['val'] = evaluate(model, s_val, t_val, device)
    results['test'] = evaluate(model, s_test, t_test, device)

    with open(save_dir / 'results.yaml', 'w') as f:
        yaml.dump(results, f, default_flow_style=False)

if __name__ == '__main__':
    main()
