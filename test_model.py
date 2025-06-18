#!/usr/bin/env python3


import argparse
from config import get_config, DEVICE
from datasets.registry import build_dataset
from models.registry import build_model
from test import test_model
from tokenizer import Tokenizer
from torch.utils.data import DataLoader


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
    dataset = build_dataset(cfg=cfg, tokenizer=tokenizer)['test']

    # dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.LOADER.TEST.BATCH_SIZE,
        shuffle=cfg.LOADER.TEST.SHUFFLE,
        num_workers=cfg.LOADER.TEST.NUM_WORKERS,
        collate_fn=dataset.collate_fn,
        pin_memory=cfg.LOADER.TEST.PIN_MEMORY,
    )

    # model
    model = build_model(cfg=cfg, tokenizer=tokenizer)
    print(model)

    test_model(
        model=model,
        ckpt_filepath=cfg.CKPT.BEST,
        dataloader=dataloader,
        device=DEVICE,
        seq_len=cfg.MODEL.SEQ2SEQ.TGT_SEQ_LEN,
        tokenizer=tokenizer,
    )


if __name__ == '__main__':
    main()
