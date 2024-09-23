from runner.options import RunnerOptions

import torch

import argparse
import importlib
import yaml
from pathlib import Path
from pprint import pprint

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str)

    parser.add_argument('--config', type=str, default='./hparams/mnist2usps.yaml')
    parser.add_argument('--checkpoint', type=str)

    parser.add_argument('--save_root', type=str, default='./runs')
    parser.add_argument('--save_dir', type=str)

    parser.add_argument('--verbose', action='store_true')

    sps = parser.add_subparsers(dest='subcmd', required=True)
    sp = sps.add_parser('train')
    sp = sps.add_parser('eval')

    args = parser.parse_args()

    if args.save_dir is None:
        save_root = Path(args.save_root)
        save_root.mkdir(parents=True, exist_ok=True)
        i = 1
        while (save_root/f'run{i}').exists():
            i += 1
        save_dir = save_root/f'run{i}'
        save_dir.mkdir()
    else:
        save_dir = Path(args.save_dir)
    
    print(f'save_dir={save_dir.absolute()}')

    with open(args.config, 'r') as f:
        options = yaml.safe_load(f)
    
    options = RunnerOptions(**options)
    pprint(options.to_dict())

    # find the corresponding module and load the needed classes
    model_mod = importlib.import_module(f'bench_model.{args.model}')
    Model = model_mod.Model
    Trainer = model_mod.Trainer
    Evaluator = model_mod.Evaluator

    # initialize model
    model = Model(hparams=options.hparams, **options.model.to_dict())

    checkpoint = options.checkpoint or args.checkpoint
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))

    # initialize runner
    if args.subcmd == 'train':
        options.checkpoint = args.checkpoint
        runner = Trainer(
            model,
            save_dir,
            progbar=True,
            options=options
        )
    
    elif args.subcmd == 'eval':
        options.checkpoint = args.checkpoint
        runner = Evaluator(
            model,
            save_dir,
            progbar=True,
            options=options
        )

    else:
        parser.print_help()
        raise ValueError(f'invalid subcmd={args.subcmd}')
    
    # load dataset
    import dataset.utils

    name = options.dataset.src.cls + '2' + options.dataset.tgt.cls
    s_kwargs = options.dataset.src.get('kwargs', None)
    t_kwargs = options.dataset.tgt.get('kwargs', None)

    if s_kwargs is not None:
        s_kwargs = s_kwargs.to_dict()
    if t_kwargs is not None:
        t_kwargs = t_kwargs.to_dict()

    loader = dataset.utils.create_uda_dataset(
        name,
        s_kwargs,
        t_kwargs,
        image_size=options.model.in_shape[1:],
        batch_size=options.batch_size,
        val_split=options.dataset.val_split
    )
    runner.set_loaders(*loader.get_loaders())

    # execute runner
    runner.execute()

    # print resulted stats
    if args.subcmd == 'train':
        pprint(runner.test_stats)
        pprint(runner.test_stats_bm)
    
    elif args.subcmd == 'eval':
        pprint(runner.cum_stats)

if __name__ == '__main__':
    main()
