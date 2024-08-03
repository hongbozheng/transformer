#!/usr/bin/env python3


from torch import Tensor
from typing import List

import argparse
from config import get_config, DEVICE
import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import IR
from logger import timestamp
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer import Transformer


def embedding(
        model: nn.Module,
        device: torch.device,
        ckpt_filepath: str,
        data_loader: DataLoader,
) -> Tensor:
    logger.log_info("Generate expression embeddings...")
    model.to(device=device)
    model.eval()
    ckpt = torch.load(f=ckpt_filepath, map_location=device)
    model.load_state_dict(state_dict=ckpt["model"])

    embs = []
    loader_tqdm = tqdm(iterable=data_loader)
    loader_tqdm.set_description(desc=f"[{timestamp()}] [Batch 0]", refresh=True)
    for i, batch in enumerate(loader_tqdm):
        src = batch["src"].to(device=device)
        src_mask = batch["src_mask"].to(device=device)
        emb = model.encode(x=src, mask=src_mask)
        emb, _ = emb.max(dim=1, keepdim=False)
        embs.append(emb.detach().cpu())
        loader_tqdm.set_description(
            desc=f"[{timestamp()}] [Batch {i+1}]",
            refresh=True,
        )
    embs = torch.cat(tensors=embs, dim=0)
    logger.log_info("Finish generating expression embeddings")

    return embs


def emb_ir(embs: Tensor, gt: List[int]) -> None:
    embs = embs.view(-1, 6, 512)

    queries = embs[:, 0, :]
    candidates = embs[:, 1:, :]

    cos_sim = cos_sim = F.cosine_similarity(
        x1=queries.unsqueeze(dim=1),
        x2=candidates,
        dim=-1,
    )

    preds = torch.argmax(input=cos_sim, dim=1)
    gt = torch.tensor(data=gt, dtype=torch.int64)
    corrects = torch.eq(input=preds, other=gt).sum().item()

    logger.log_info(f"Accuracy {corrects/gt.size(dim=0)*100:.4f}%")

    return


def main() -> None:
    cfg = get_config(args=None)

    parser = argparse.ArgumentParser(
        prog="emb_ir",
        description="Get embeddings of mathematical expressions "
                    "and perform embedding information retriveal"
    )
    parser.add_argument(
        "--filepath",
        "-f",
        type=str,
        required=True,
        help="embedding ir filepath",
    )

    args = parser.parse_args()
    filepath = args.filepath

    tokenizer = Tokenizer()

    ir = IR(filepath=filepath, tokenizer=tokenizer)

    ir_loader = DataLoader(
        dataset=ir,
        batch_size=cfg.LOADER.VAL.BATCH_SIZE,
        shuffle=cfg.LOADER.VAL.SHUFFLE,
        num_workers=cfg.LOADER.VAL.NUM_WORKERS,
        collate_fn=ir.collate_fn,
        pin_memory=cfg.LOADER.VAL.PIN_MEMORY,
    )

    model = Transformer(
        emb_dim=cfg.MODEL.TX.EMB_DIM,
        src_vocab_size=len(tokenizer.components),
        tgt_vocab_size=len(tokenizer.components),
        src_seq_len=cfg.MODEL.TX.SRC_SEQ_LEN,
        tgt_seq_len=cfg.MODEL.TX.TGT_SEQ_LEN,
        n_encoder_layers=cfg.MODEL.TX.N_ENCODER_LAYERS,
        n_decoder_layers=cfg.MODEL.TX.N_DECODER_LAYERS,
        n_heads=cfg.MODEL.TX.N_HEADS,
        dropout=cfg.MODEL.TX.DROPOUT,
        dim_feedforward=cfg.MODEL.TX.DIM_FEEDFORWARD,
    )

    embs = embedding(
        model=model,
        device=DEVICE,
        ckpt_filepath=cfg.BEST_MODEL.TX,
        data_loader=ir_loader,
    )

    emb_ir(embs=embs, gt=ir.gt)

    return


if __name__ == "__main__":
    main()
