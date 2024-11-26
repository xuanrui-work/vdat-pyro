from .options import RunnerOptions

import torch
import torch.nn as nn

import yaml
from pathlib import Path

class BaseRunner:
    def __init__(
        self,
        model: nn.Module,
        save_dir: str = '',
        options: RunnerOptions|dict = None
    ):
        if isinstance(options, dict):
            options = RunnerOptions(**options)
        elif options is None:
            options = RunnerOptions()
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            with open(save_dir/'options.yml', 'w') as f:
                yaml.safe_dump(options.to_dict(), f)

        self.model = model
        self.save_dir = save_dir
        self.options = options

        self.device = options.device
        self.batch_size = options.batch_size
    
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))
    
    def save_stats(self, path: str, stats: dict):
        with open(path, 'w') as f:
            yaml.safe_dump(stats, f, default_flow_style=True)
    
    def load_stats(self, path: str):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def before_run(self):
        pass

    def after_run(self):
        pass

    def run(self):
        raise NotImplementedError
    
    def execute(self):
        self.before_run()
        self.run()
        self.after_run()
