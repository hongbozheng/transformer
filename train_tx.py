#!/usr/bin/env python3


from config import get_config, DEVICE
from criterion import build_criterion
from dataset import CL
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from train import train_model
from transformer import Transformer


def main() -> None:
    cfg = get_config(args=None)

    tokenizer = Tokenizer()

    cl_dataset = CL(
        filepath=cfg.DATA.TRAIN_FILE,
        tokenizer=tokenizer,
    )

    dataloader = DataLoader(
        dataset=cl_dataset,
        batch_size=cfg.LOADER.TRAIN.BATCH_SIZE,
        shuffle=cfg.LOADER.TRAIN.SHUFFLE,
        num_workers=cfg.LOADER.TRAIN.NUM_WORKERS,
        collate_fn=cl_dataset.collate_fn,
        pin_memory=cfg.LOADER.TRAIN.PIN_MEMORY,
    )
    # for batch in train_loader:
    #     print(batch["src"], batch["src"].shape)
    #     print(batch["src_mask"], batch["src_mask"].shape)
    #     # print(batch["tgt"], batch["tgt"].shape)
    #     break
    # exit()
    # val_loader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=cfg.LOADER.VAL.BATCH_SIZE,
    #     shuffle=cfg.LOADER.VAL.SHUFFLE,
    #     num_workers=cfg.LOADER.VAL.NUM_WORKERS,
    #     collate_fn=val_dataset.collate_fn,
    #     pin_memory=cfg.LOADER.VAL.PIN_MEMORY,
    # )

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

    # define criterion
    criterion = build_criterion(cfg=cfg)

    # criterion = InfoNCE(
    #     temperature=cfg.CRITERION.INFONCE.TEMPERATURE,
    #     reduction=cfg.CRITERION.INFONCE.REDUCTION,
    # )

    # # criterion = SimCSE(
    # #     temperature=cfg.CRITERION.INFONCE.TEMPERATURE,
    # #     reduction=cfg.CRITERION.INFONCE.REDUCTION,
    # # )

    # # criterion = ContrastiveLoss(
    # #     margin=cfg.CRITERION.CL.MARGIN,
    # #     reduction=cfg.CRITERION.CL.REDUCTION,
    # # )

    train_model(
        model=model,
        ckpt_best=cfg.CKPT.BEST,
        ckpt_last=cfg.CKPT.LAST,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        max_norm=cfg.TRAIN.MAX_NORM,
        device=DEVICE,
        n_epochs=cfg.TRAIN.N_EPOCHS,
        dataloader=dataloader,
        save_every_n_iters=cfg.TRAIN.SAVE_N_ITERS,
    )


if __name__ == '__main__':
    main()
