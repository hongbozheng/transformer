#!/usr/bin/env python3


from torch import Tensor
from typing import List

import argparse
from config import get_config, DEVICE, SEED
import matplotlib.pyplot as plt
import numpy as np
from dataset import KMC
from emb import embedding
from sklearn.manifold import TSNE
from tokenizer import Tokenizer
from tokenizer_sememb import TokenizerSemEmb
from torch.utils.data import DataLoader
from transformer import Transformer
from umap import UMAP


colors = ['#377eb8', '#ff7f00', '#4daf4a',
          '#f781bf', '#a65628', '#984ea3',
          '#999999', '#e41a1c', '#dede00',]


def emb_plt(
        method: str,
        eggemb_embs: Tensor,
        sememb_embs: Tensor,
        perplexity: int,
        gt: List[int],
) -> None:
    if method == "UMAP":
        reducer = UMAP(n_neighbors=perplexity, n_components=2, transform_seed=SEED)
        reducer.fit(X=eggemb_embs)
        eggemb_embs = reducer.transform(X=eggemb_embs)
        reducer.fit(X=sememb_embs)
        sememb_embs = reducer.transform(X=sememb_embs)
    elif method == "t-SNE":
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=SEED)
        eggemb_embs = tsne.fit_transform(X=eggemb_embs)
        sememb_embs = tsne.fit_transform(X=sememb_embs)

    plt.rc(group="font", family="serif")
    plt.rc(group="text", usetex=True)

    classes = np.unique(ar=gt)
    colors = plt.cm.tab20.colors[:len(classes)]
    labels = [
        r"$\ln x$",
        r"$\sin x$", r"$\cos x$", r"$\tan x$",
        r"$\csc x$", r"$\sec x$", r"$\cot x$",
        r"$\sin^{-1} x$", r"$\cos^{-1} x$", r"$\tan^{-1} x$",
        r"$\sinh x$", r"$\cosh x$", r"$\tanh x$",
        r"$\coth x$",
        r"$\sinh^{-1} x$", r"$\cosh^{-1} x$", r"$\tanh^{-1} x$",
    ]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
    for i, cls in enumerate(classes):
        id = cls == gt
        ax[0].scatter(
            x=eggemb_embs[id, 0],
            y=eggemb_embs[id, 1],
            color=colors[i % len(colors)],
            label=labels[i],
            s=5,
        )
    ax[0].set_xlabel('Component 1', fontsize=26, fontweight=2)
    ax[0].set_ylabel('Component 2', fontsize=26, fontweight=2)
    ax[0].tick_params(axis='both', which='major', labelsize=26)
    ax[0].spines["top"].set_visible(b=False)
    ax[0].spines["bottom"].set_visible(b=True)
    ax[0].spines["left"].set_visible(b=True)
    ax[0].spines["right"].set_visible(b=False)

    for i, cls in enumerate(classes):
        id = cls == gt
        ax[1].scatter(
            x=sememb_embs[id, 0],
            y=sememb_embs[id, 1],
            color=colors[i % len(colors)],
            label=labels[i],
            s=5,
        )
    ax[1].set_xlabel('Component 1', fontsize=26, fontweight=2)
    ax[1].set_ylabel('Component 2', fontsize=26, fontweight=2)
    ax[1].tick_params(axis='both', which='major', labelsize=26)
    ax[1].spines["top"].set_visible(b=False)
    ax[1].spines["bottom"].set_visible(b=True)
    ax[1].spines["left"].set_visible(b=True)
    ax[1].spines["right"].set_visible(b=False)

    legend = [
        plt.Line2D(
            xdata=[],
            ydata=[],
            color=color,
            lw=0,
            marker='s',
            markersize=24,
        ) for color in colors
    ]

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles=legend,
        labels=labels,
        loc=8,
        ncols=9,
        fontsize=28,
        frameon=False,
        handletextpad=0.1,
        # borderaxespad=0.1,
        columnspacing=0.8,

    )

    plt.tight_layout(rect=[0, 0.150 , 1, 1])
    plt.savefig(f"{method}.svg", transparent=True, dpi=500, format="svg")
    # plt.savefig(f"{method}.png", transparent=False, dpi=100)

    return


def main() -> None:
    cfg = get_config(args=None)

    parser = argparse.ArgumentParser(
        prog="kmeans",
        description="1x2 K-Means plot"
    )
    parser.add_argument(
        "--eggemb",
        "-m0",
        type=str,
        required=True,
        help="eggemb checkpoint filepath",
    )
    parser.add_argument(
        "--sememb",
        "-m1",
        type=str,
        required=True,
        help="sememb checkpoint filepath",
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
        required=True,
        choices=["UMAP", "t-SNE"],
        help="dimension reduction method",
    )

    args = parser.parse_args()
    eggemb_ckpt = args.eggemb
    sememb_ckpt = args.sememb
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
    eggemb = Transformer(
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

    tokenizer_sememb = TokenizerSemEmb()
    kmc_sememb = KMC(filepath=filepath, tokenizer=tokenizer_sememb)
    kmc_loader_sememb = DataLoader(
        dataset=kmc_sememb,
        batch_size=cfg.LOADER.VAL.BATCH_SIZE,
        shuffle=cfg.LOADER.VAL.SHUFFLE,
        num_workers=cfg.LOADER.VAL.NUM_WORKERS,
        collate_fn=kmc_sememb.collate_fn,
        pin_memory=cfg.LOADER.VAL.PIN_MEMORY,
    )
    sememb = Transformer(
        emb_dim=cfg.MODEL.TX.EMB_DIM,
        src_vocab_size=len(tokenizer_sememb.components),
        tgt_vocab_size=len(tokenizer_sememb.components),
        src_seq_len=cfg.MODEL.TX.SRC_SEQ_LEN,
        tgt_seq_len=cfg.MODEL.TX.TGT_SEQ_LEN,
        n_encoder_layers=cfg.MODEL.TX.N_ENCODER_LAYERS,
        n_decoder_layers=cfg.MODEL.TX.N_DECODER_LAYERS,
        n_heads=cfg.MODEL.TX.N_HEADS,
        dropout=cfg.MODEL.TX.DROPOUT,
        dim_feedforward=cfg.MODEL.TX.DIM_FEEDFORWARD,
    )

    eggemb_embs = embedding(
        model=eggemb,
        device=DEVICE,
        ckpt_filepath=eggemb_ckpt,
        data_loader=kmc_loader,
        mode=mode,
    )

    sememb_embs = embedding(
        model=sememb,
        device=DEVICE,
        ckpt_filepath=sememb_ckpt,
        data_loader=kmc_loader_sememb,
        mode=mode,
    )

    emb_plt(
        method=method,
        eggemb_embs=eggemb_embs,
        sememb_embs=sememb_embs,
        perplexity=cfg.DIM_RED.PERPLEXITY,
        gt=kmc.gt,
    )

    return


if __name__ == "__main__":
    main()
