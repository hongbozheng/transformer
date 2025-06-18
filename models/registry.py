import torch.nn as nn


MODEL_REGISTRY = {}


def register_model(name):
    def decorator(fn):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model `{name}` is already registered")
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def build_model(cfg, tokenizer) -> nn.Module:
    name = cfg.MODEL.NAME
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model `{name}` not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](cfg=cfg, tokenizer=tokenizer)
