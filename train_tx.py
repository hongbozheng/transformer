#!/usr/bin/env python3


import torch.nn as nn
from config import get_config, DEVICE
from dataset import EquivExpr
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from train import train_model
from transformer import Transformer


def main() -> None:
    cfg = get_config(args=None)

    tokenizer = Tokenizer()

    train_dataset = EquivExpr(
        filepath=cfg.DATA.TRAIN_FILE,
        tokenizer=tokenizer,
        val=False,
    )
    val_dataset = EquivExpr(
        filepath=cfg.DATA.VAL_FILE,
        tokenizer=tokenizer,
        val=True,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.LOADER.TRAIN.BATCH_SIZE,
        shuffle=cfg.LOADER.TRAIN.SHUFFLE,
        num_workers=cfg.LOADER.TRAIN.NUM_WORKERS,
        collate_fn=train_dataset.collate_fn,
        pin_memory=cfg.LOADER.TRAIN.PIN_MEMORY,
    )
    # for batch in train_loader:
    #     print(batch["src"], batch["src"].shape)
    #     print(batch["tgt"], batch["tgt"].shape)
    #     print(batch["src_mask"])
    #     print(batch["tgt_mask"])
    #     break
    # return
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.LOADER.VAL.BATCH_SIZE,
        shuffle=cfg.LOADER.VAL.SHUFFLE,
        num_workers=cfg.LOADER.VAL.NUM_WORKERS,
        collate_fn=val_dataset.collate_fn,
        pin_memory=cfg.LOADER.VAL.PIN_MEMORY,
    )

    model = Transformer(
        dim=cfg.MODEL.TX.DIM,
        src_vocab_size=len(tokenizer.symbols),
        tgt_vocab_size=len(tokenizer.symbols),
        src_seq_len=cfg.MODEL.TX.SRC_SEQ_LEN,
        tgt_seq_len=cfg.MODEL.TX.TGT_SEQ_LEN,
        n_encoder_layers=cfg.MODEL.TX.N_ENCODER_LAYERS,
        n_decoder_layers=cfg.MODEL.TX.N_DECODER_LAYERS,
        n_heads=cfg.MODEL.TX.N_HEADS,
        dropout=cfg.MODEL.TX.DROPOUT,
        dim_feedforward=cfg.MODEL.TX.DIM_FEEDFORWARD,
    )

    # define optimizer
    optimizer = build_optimizer(cfg=cfg, model=model)

    # define lr scheduler
    lr_scheduler = build_scheduler(cfg=cfg, optimizer=optimizer)

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.sym2idx["PAD"],
        label_smoothing=cfg.CRITERION.CROSSENTROPY.LABEL_SMOOTHING,
    )

    train_model(
        model=model,
        device=DEVICE,
        ckpt_best=cfg.CKPT.BEST,
        ckpt_last=cfg.CKPT.LAST,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        n_epochs=cfg.TRAIN.N_EPOCHS,
        criterion=criterion,
        max_norm=cfg.TRAIN.MAX_NORM,
        train_loader=train_loader,
        val_loader=val_loader,
        seq_len=cfg.MODEL.TX.TGT_SEQ_LEN,
        tokenizer=tokenizer,
        save_every_n_iters=cfg.TRAIN.SAVE_N_ITERS,
    )

    return


if __name__ == '__main__':
    main()
