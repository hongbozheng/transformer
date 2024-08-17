#!/usr/bin/env python3


from torch import Tensor
from typing import List

import argparse
from config import get_config, DEVICE
import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import EA
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
    logger.log_info(f"Loaded model '{ckpt_filepath}'")

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


def emb_algebra(embs: Tensor, exprs: List[str], filepath: str) -> None:
    file = open(file=filepath, mode='r', encoding="utf-8")
    expr_tuple = [line.strip() for line in file.readlines()]
    file.close()

    res_file = open(file="data/emb_alg_res.txt", mode='w', encoding="utf-8")

    eval_tqdm = tqdm(iterable=expr_tuple)
    eval_tqdm.set_description(desc=f"[{timestamp()}] [Batch 0]", refresh=True)

    corrects = 0
    incorrects = 0

    for line in eval_tqdm:
        expr = line.strip().split(sep='\t')

        eval_tqdm.set_description(
            desc=f"[{timestamp()}] [INFO]: Processing '{expr[0]}' '{expr[1]}' "
                 f"'{expr[2]}' '{expr[3]}'",
            refresh=True,
        )

        id0 = exprs.index(expr[0])
        id1 = exprs.index(expr[1])
        id2 = exprs.index(expr[2])
        gt = exprs.index(expr[3])
        indices = [id0, id1, id2]

        pred_emb = -embs[id0] + embs[id1] + embs[id2]
        cos_sim = F.cosine_similarity(
            x1=pred_emb.unsqueeze(dim=0),
            x2=embs,
            dim=1,
        )
        _, pred_indices = torch.topk(input=cos_sim, k=4)

        if pred_indices[0] not in indices:
            pred = pred_indices[0]
        elif pred_indices[1] not in indices:
            pred = pred_indices[1]
        elif pred_indices[2] not in indices:
            pred = pred_indices[2]
        else:
            pred = pred_indices[3]

        if pred == gt:
            res_file.write(f"{expr[0]}\t{expr[1]}\t{expr[2]}\t{exprs[pred]}\n")
            corrects += 1
        else:
            res_file.write(f"{expr[0]}\t{expr[1]}\t{expr[2]}\t{exprs[pred]}\t[{exprs[gt]}]\n")
            incorrects += 1

    res_file.close()

    logger.log_info(f"Accuracy {corrects/(corrects+incorrects)*100:.4f}%")

    return


def main() -> None:
    cfg = get_config(args=None)

    parser = argparse.ArgumentParser(
        prog="emb_algebra",
        description="Get embeddings of mathematical expressions "
                    "and perform embedding algebra"
    )
    parser.add_argument(
        "--ckpt_filepath",
        "-m",
        type=str,
        required=True,
        help="model checkpoint filepath",
    )
    parser.add_argument(
        "--pool",
        "-p",
        type=str,
        required=True,
        help="expressions pool filepath",
    )
    parser.add_argument(
        "--filepath",
        "-f",
        type=str,
        required=True,
        help="embedding algebra filepath",
    )

    args = parser.parse_args()
    ckpt_filepath = args.ckpt_filepath
    pool = args.pool
    filepath = args.filepath

    tokenizer = Tokenizer()

    ea = EA(filepath=pool, tokenizer=tokenizer)

    ea_loader = DataLoader(
        dataset=ea,
        batch_size=cfg.LOADER.VAL.BATCH_SIZE,
        shuffle=cfg.LOADER.VAL.SHUFFLE,
        num_workers=cfg.LOADER.VAL.NUM_WORKERS,
        collate_fn=ea.collate_fn,
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
        ckpt_filepath=ckpt_filepath,
        data_loader=ea_loader,
    )

    emb_algebra(embs=embs, exprs=ea.exprs, filepath=filepath)

    return


if __name__ == "__main__":
    main()
