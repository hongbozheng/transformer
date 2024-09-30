#!/usr/bin/env python3


from torch import Tensor
from typing import List

import argparse
from config import get_config, DEVICE, SEED
import logger
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from dataset import KMC
from emb import embedding
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from transformer import Transformer
from umap import UMAP


def calculate_acc(
        n_clusters: int,
        len_cluster: List[int],
        rest_labels: List[int],
) -> None:
    all_labels = {i for i in range(n_clusters)}
    used_labels = []
    accs = []
    hard_clusters = []

    for i in range(n_clusters):
        labels = rest_labels[:len_cluster[i]]
        if i < n_clusters:
            rest_labels = rest_labels[len_cluster[i]:]
        j = 1
        while j <= n_clusters:
            if j <= len(Counter(labels).most_common(j)):
                most_label = Counter(labels).most_common(j)[j-1][0]
                if most_label not in used_labels:
                    used_labels.append(most_label)
                    accs.append(labels.count(most_label)/len(labels))
                    break
                else:
                    j += 1
            else:
                hard_clusters.append(labels)
                break
    hard_labels = list(all_labels - set(used_labels))
    for i in range(len(hard_clusters)):
        most_label = hard_labels[i]
        labels = hard_clusters[i]
        accs.append(labels.count(most_label)/len(labels))

    acc = np.mean(a=accs)
    logger.log_info(f"Accuracy {acc*100:.4f}%")

    return


def emb_plt(method: str, embs: Tensor, perplexity: int, gt: List[int]) -> None:
    if method == "UMAP":
        reducer = UMAP(n_neighbors=perplexity, n_components=2, transform_seed=SEED)
        reducer.fit(X=embs)
        embs = reducer.transform(X=embs)
    elif method == "t-SNE":
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=SEED)
        embs = tsne.fit_transform(X=embs)

    plt.rc(group="font", family="serif")
    plt.rc(group="text", usetex=False)

    colors = ['#377eb8', '#ff7f00', '#4daf4a',
              '#f781bf', '#a65628', '#984ea3',
              '#999999', '#e41a1c', '#dede00',]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    ax.scatter(x=embs[:, 0], y=embs[:, 1], c=gt, cmap="Spectral", s=5)
    ax.spines["top"].set_visible(b=False)
    ax.spines["bottom"].set_visible(b=False)
    ax.spines["left"].set_visible(b=False)
    ax.spines["right"].set_visible(b=False)

    plt.tight_layout()
    plt.savefig(f"{method}.svg", transparent=True, dpi=100, format="svg")

    return


def main() -> None:
    cfg = get_config(args=None)

    parser = argparse.ArgumentParser(
        prog="emb_cluster",
        description="Get embeddings of mathematical expressions "
                    "and perform KMeans-Clustering"
    )
    parser.add_argument(
        "--ckpt_filepath",
        "-m",
        type=str,
        required=True,
        help="model checkpoint filepath",
    )
    parser.add_argument(
        "--mode",
        "-e",
        type=str,
        required=True,
        choices=["mean", "max"],
        help="embedding mode",
    )
    parser.add_argument(
        "--filepath",
        "-f",
        type=str,
        required=True,
        help="expressions filepath",
    )
    parser.add_argument(
        "--dim_red",
        "-d",
        type=str,
        required=False,
        choices=["UMAP", "t-SNE"],
        help="dimension reduction method",
    )

    args = parser.parse_args()
    ckpt_filepath = args.ckpt_filepath
    mode = args.mode
    filepath = args.filepath
    method = args.dim_red

    tokenizer = Tokenizer()

    kmc = KMC(filepath=filepath, tokenizer=tokenizer)

    kmc_loader = DataLoader(
        dataset=kmc,
        batch_size=cfg.LOADER.VAL.BATCH_SIZE,
        shuffle=cfg.LOADER.VAL.SHUFFLE,
        num_workers=cfg.LOADER.VAL.NUM_WORKERS,
        collate_fn=kmc.collate_fn,
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
        data_loader=kmc_loader,
        mode=mode,
    )
    embs = embs.numpy()

    kmeans = KMeans(
        n_clusters=kmc.n_clusters,
        max_iter=cfg.KMEANS.MAX_ITER,
        tol=cfg.KMEANS.TOL,
        random_state=cfg.KMEANS.RANDOM_STATE,
    )
    kmeans.fit(X=embs)
    kmeans.predict(X=embs)

    calculate_acc(
        n_clusters=kmc.n_clusters,
        len_cluster=kmc.sizes,
        rest_labels=list(kmeans.labels_),
    )

    if method:
        emb_plt(
            method=method,
            embs=embs,
            perplexity=cfg.DIM_RED.PERPLEXITY,
            gt=kmc.gt,
        )

    return


if __name__ == "__main__":
    main()
