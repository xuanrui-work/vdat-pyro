import yaml

class RunnerOptions:
    r"""
    Options for a runner stored in a class as an alternative to using a dictionary.
    Options can be accessed as class attributes. E.g. `options.batch_size`.
    """

    def __init__(self, **kwargs):
        self.fields = {}
        for k, v in kwargs.items():
            # print(k, v)
            if isinstance(v, dict):
                self.fields[k] = RunnerOptions(**v)
            else:
                self.fields[k] = v
    
    def __getattr__(self, name: str):
        try:
            return self.fields[name]
        except KeyError as err:
            raise AttributeError(name) from err

    def __setattr__(self, name: str, value: any):
        super().__setattr__(name, value)
        if name != 'fields':
            self.fields[name] = value
    
    def __getitem__(self, name: str):
        return self.fields[name]
    
    def __setitem__(self, name: str, value: any):
        self.fields[name] = value
    
    def __iter__(self):
        return iter(self.fields)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.to_dict()})'
    
    def get(self, name: str, default: any = None):
        return self.fields.get(name, default)
    
    def to_dict(self) -> dict:
        d = {}
        for k, v in self.fields.items():
            d[k] = v.to_dict() if isinstance(v, RunnerOptions) else v
        return d

    def save(self, path: str):
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f)
    
    @staticmethod
    def load(path: str) -> 'RunnerOptions':
        with open(path, 'r') as f:
            d = yaml.safe_load(f)
        return RunnerOptions(**d)
