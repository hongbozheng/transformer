#!/usr/bin/env python3


from torch import Tensor
from typing import List

import argparse
from config import get_config, DEVICE
import logger
import torch
import torch.nn.functional as F
from dataset import IR
from emb import embedding
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from transformer import Transformer


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
        "--ckpt_filepath",
        "-m",
        type=str,
        required=True,
        help="model checkpoint filepath",
    )
    parser.add_argument(
        "--filepath",
        "-f",
        type=str,
        required=True,
        help="embedding ir filepath",
    )
    parser.add_argument(
        "--mode",
        "-e",
        type=str,
        required=True,
        choices=["mean", "max"],
        help="embedding mode",
    )

    args = parser.parse_args()
    ckpt_filepath = args.ckpt_filepath
    filepath = args.filepath
    mode = args.mode

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
        ckpt_filepath=ckpt_filepath,
        data_loader=ir_loader,
        mode=mode,
    )

    emb_ir(embs=embs, gt=ir.gt)

    return


if __name__ == "__main__":
    main()
