from torch.utils.data import Dataset
from typing import Dict


DATA_REGISTRY = {}


def register_dataset(name: str):
    def decorator(fn):
        if name in DATA_REGISTRY:
            raise ValueError(f"Dataset name `{name}` already registered")
        DATA_REGISTRY[name] = fn
        return fn
    return decorator


def build_dataset(cfg, tokenizer) -> Dict[str, Dataset]:
    name = cfg.DATA.NAME
    if name not in DATA_REGISTRY:
        raise KeyError(f"Dataset `{name}` not found in registry. Available: {list(DATA_REGISTRY.keys())}")
    return DATA_REGISTRY[name](cfg=cfg, tokenizer=tokenizer)
