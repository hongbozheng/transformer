#!/usr/bin/env python3


from config import get_config, DEVICE
from tokenizer import Tokenizer
from dataset import CL
from torch.utils.data import DataLoader
from transformer import Transformer
from train import train_model
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from criterion import InfoNCE


def main() -> None:
    cfg = get_config(args=None)

    tokenizer = Tokenizer()

    cl_dataset = CL(
        filepath=cfg.DATA.TRAIN_FILE,
        tokenizer=tokenizer,
    )

    train_loader = DataLoader(
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

    optimizer = Adam(
        params=model.parameters(),
        lr=cfg.MODEL.TX.LR,
        weight_decay=cfg.MODEL.TX.WEIGHT_DECAY,
    )

    lr_scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=10,
        eta_min=1e-8,
        last_epoch=-1,
    )

    criterion = InfoNCE(
        temperature=cfg.TRAIN.TEMPERATURE,
        reduction=cfg.TRAIN.REDUCTION,
    )

    train_model(
        model=model,
        device=DEVICE,
        ckpt_filepath=cfg.BEST_MODEL.TX,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        n_epochs=cfg.TRAIN.N_EPOCHS,
        criterion=criterion,
        max_norm=cfg.TRAIN.MAX_NORM,
        train_loader=train_loader,
    )

    return


if __name__ == '__main__':
    main()
