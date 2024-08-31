from torch import Tensor

import logger
import torch
import torch.nn as nn
from logger import timestamp
from torch.utils.data import DataLoader
from tqdm import tqdm


def embedding(
        model: nn.Module,
        device: torch.device,
        ckpt_filepath: str,
        data_loader: DataLoader,
        mode: str,
) -> Tensor:
    logger.log_info("Generate expression embeddings...")
    model.to(device=device)
    model.eval()
    ckpt = torch.load(f=ckpt_filepath, map_location=device)
    model.load_state_dict(state_dict=ckpt["model"])
    logger.log_info(f"Loaded model '{ckpt_filepath}'")

    embs = []
    loader_tqdm = tqdm(iterable=data_loader)
    loader_tqdm.set_description(desc=f"[{timestamp()}] [Batch 0]", refresh=True)
    for i, batch in enumerate(loader_tqdm):
        src = batch["src"].to(device=device)
        src_mask = batch["src_mask"].to(device=device)
        emb = model.encode(x=src, mask=src_mask)

        src_mask = src_mask.squeeze(dim=(-3, -2))
        src_mask[:, 0] = 0
        last_1 = src_mask.sum(dim=1)
        src_mask[torch.arange(src_mask.size(dim=0)), last_1] = 0

        if mode == "mean":
            emb[src_mask==0] = 0
            emb = emb.sum(dim=-2)
            last_1 = last_1.unsqueeze(dim=1)
            emb /= last_1
        elif mode == "max":
            emb[src_mask==0] = float("-inf")
            emb, _ = emb.max(dim=1, keepdim=False)
        else:
            logger.log_error("Invalid mode for embedding!")
        
        embs.append(emb.detach().cpu())
        loader_tqdm.set_description(
            desc=f"[{timestamp()}] [Batch {i+1}]",
            refresh=True,
        )
    embs = torch.cat(tensors=embs, dim=0)
    logger.log_info("Finish generating expression embeddings")

    return embs
