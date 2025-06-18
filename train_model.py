#!/usr/bin/env python3


import argparse
from config import get_config, DEVICE
from criterion import build_criterion
from datasets.registry import build_dataset
from lr_scheduler import build_scheduler
from models.registry import build_model
from optimizer import build_optimizer
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='semantic representations of mathematical expressions'
    )
    parser.add_argument(
        '--cfg',
        type=str,
        required=True,
        metavar="FILE",
        help='path to config file',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        metavar="FILE",
        help='path to dataset config file',
    )
    args, unparsed = parser.parse_known_args()
    cfg = get_config(args=args)

    tokenizer = Tokenizer()

    # dataset
    train_dataset = build_dataset(cfg=cfg, tokenizer=tokenizer)['train']
    val_dataset = build_dataset(cfg=cfg, tokenizer=tokenizer)['val']

    # dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.LOADER.TRAIN.BATCH_SIZE,
        shuffle=cfg.LOADER.TRAIN.SHUFFLE,
        num_workers=cfg.LOADER.TRAIN.NUM_WORKERS,
        collate_fn=train_dataset.collate_fn,
        pin_memory=cfg.LOADER.TRAIN.PIN_MEMORY,
    )

    if val_dataset is not None:
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=cfg.LOADER.VAL.BATCH_SIZE,
            shuffle=cfg.LOADER.VAL.SHUFFLE,
            num_workers=cfg.LOADER.VAL.NUM_WORKERS,
            collate_fn=val_dataset.collate_fn,
            pin_memory=cfg.LOADER.VAL.PIN_MEMORY,
        )
    else:
        val_loader = None

    # model
    model = build_model(cfg=cfg, tokenizer=tokenizer)
    print(model)

    # optimizer
    optimizer = build_optimizer(cfg=cfg, model=model)

    # lr scheduler
    lr_scheduler = build_scheduler(cfg=cfg, optimizer=optimizer)

    # criterion
    criterion = build_criterion(cfg=cfg, ignore_index=tokenizer.sym2idx["PAD"])

    train_model(
        model=model,
        ckpt_best=cfg.CKPT.BEST,
        ckpt_last=cfg.CKPT.LAST,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        postprocess=cfg.POSTPROCESS.NAME,
        n_exprs=cfg.DATA.N_EXPRS,
        criterion=criterion,
        max_norm=cfg.TRAIN.MAX_NORM,
        device=DEVICE,
        n_epochs=cfg.TRAIN.N_EPOCHS,
        train_loader=train_loader,
        val_loader=val_loader,
        seq_len=cfg.MODEL.SEQ2SEQ.TGT_SEQ_LEN,
        tokenizer=tokenizer,
        save_every_n_iters=cfg.TRAIN.SAVE_N_ITERS,
    )


if __name__ == '__main__':
    main()
