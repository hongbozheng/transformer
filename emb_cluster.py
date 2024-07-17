#!/usr/bin/env python3


import argparse
from config import get_config, DEVICE
import logger
import torch
import torch.nn as nn
from dataset import CL_KMeans
from logger import timestamp
from sklearn.cluster import KMeans
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer import Transformer


def embedding(
        model: nn.Module,
        device: torch.device,
        ckpt_filepath: str,
        data_loader: DataLoader,
) -> torch.Tensor:
    logger.log_info("Generate expression embeddings...")
    model.to(device=device)
    model.eval()
    ckpt = torch.load(f=ckpt_filepath, map_location=device)
    model.load_state_dict(state_dict=ckpt["model"])

    embs = []
    loader_tqdm = tqdm(iterable=data_loader)
    for i, batch in enumerate(loader_tqdm):
        src = batch["src"].to(device=device)
        src_mask = batch["src_mask"].to(device=device)
        emb = model.encode(x=src, mask=src_mask)
        emb, _ = emb.max(dim=1, keepdim=False)
        embs.append(emb)
        loader_tqdm.set_description(
            desc=f"[{timestamp()}] [Batch {i+1}]",
            refresh=True
        )
    embs = torch.cat(tensors=embs, dim=0)
    logger.log_info("Finish generating expression embeddings")

    return embs


def main() -> None:
    cfg = get_config(args=None)

    parser = argparse.ArgumentParser(
        prog="emb_cluster",
        description="Get embeddings of mathematical expressions "
                    "and perform KMeans-Clustering"
    )
    parser.add_argument(
        "--filepath",
        "-f",
        type=str,
        required=True,
        help="expressions filepath"
    )

    args = parser.parse_args()
    filepath = args.filepath

    tokenizer = Tokenizer()

    cl_dataset = CL_KMeans(filepath=filepath, tokenizer=tokenizer)

    cl_loader = DataLoader(
        dataset=cl_dataset,
        batch_size=cfg.LOADER.VAL.BATCH_SIZE,
        shuffle=cfg.LOADER.TRAIN.SHUFFLE,
        num_workers=cfg.LOADER.TRAIN.NUM_WORKERS,
        collate_fn=cl_dataset.collate_fn,
        pin_memory=cfg.LOADER.TRAIN.PIN_MEMORY,
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
        data_loader=cl_loader,
    )
    embs = embs.cpu().detach().numpy()

    kmeans = KMeans(
        n_clusters=cl_dataset.n_clusters,
        max_iter=cfg.KMEANS.MAX_ITER,
        tol=cfg.KMEANS.TOL,
        random_state=cfg.KMEANS.RANDOM_STATE,
    )
    kmeans.fit(X=embs)
    kmeans.predict(X=embs)

    print(kmeans.labels_)

    return


if __name__ == "__main__":
    main()
