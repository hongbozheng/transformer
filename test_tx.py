#!/usr/bin/env python3


from config import get_config, DEVICE
from dataset import EquivExpr
from test import test_model
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from transformer import Transformer


def main() -> None:
    cfg = get_config(args=None)

    tokenizer = Tokenizer()

    test_dataset = EquivExpr(
        filepath=cfg.DATA.TEST_FILE,
        tokenizer=tokenizer,
        val=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.LOADER.TEST.BATCH_SIZE,
        shuffle=cfg.LOADER.TEST.SHUFFLE,
        num_workers=cfg.LOADER.TEST.NUM_WORKERS,
        collate_fn=test_dataset.collate_fn,
        pin_memory=cfg.LOADER.TEST.PIN_MEMORY,
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

    test_model(
        model=model,
        ckpt_filepath=cfg.BEST_MODEL.TX,
        device=DEVICE,
        test_loader=test_loader,
        seq_len=cfg.MODEL.TX.TGT_SEQ_LEN,
        tokenizer=tokenizer,
    )

    return


if __name__ == '__main__':
    main()
