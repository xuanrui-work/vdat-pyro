from .base_dataset import MultiLoader
from . import datasets

def create_uda_dataset(
    name: str,
    s_kwargs: dict = None,
    t_kwargs: dict = None,
    **kwargs
) -> MultiLoader:
    r"""Create dataset for UDA using `name`.
    Args:
        name: Name of the dataset in the format of "{src_cls}2{tgt_cls}". where `src_cls` and 
            `tgt_cls` are the names of the classes of the src and tgt datasets written in `datasets`.
        s_kwargs: Keyword arguments for the src dataset.
        t_kwargs: Keyword arguments for the tgt dataset.
        kwargs: Keyword arguments for both src and tgt datasets.
    Returns:
        The dataset for UDA in the form of a `MultiLoader`.
    """
    if s_kwargs is None:
        s_kwargs = {}
    if t_kwargs is None:
        t_kwargs = {}

    s_cls_str, t_cls_str = name.split('2')

    try:
        s_cls = getattr(datasets, s_cls_str)
    except AttributeError as err:
        raise ValueError(f'dataset "{s_cls_str}" not found') from err
    try:
        t_cls = getattr(datasets, t_cls_str)
    except AttributeError as err:
        raise ValueError(f'dataset "{t_cls_str}" not found') from err
    
    s_args = kwargs.copy()
    s_args.update(s_kwargs)
    s_dataset = s_cls(**s_args)

    t_args = kwargs.copy()
    t_args.update(t_kwargs)
    t_dataset = t_cls(**t_args)

    return MultiLoader([s_dataset, t_dataset])
