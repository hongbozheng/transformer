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
from torch.utils.data import DataLoader
from transformer import Transformer
from umap import UMAP


colors = ['#377eb8', '#ff7f00', '#4daf4a',
          '#f781bf', '#a65628', '#984ea3',
          '#999999', '#e41a1c', '#dede00',]


def emb_plt(
        method: str,
        embs: Tensor,
        perplexity: int,
        gt: List[int],
        exprs: List[str],
) -> None:
    if method == "UMAP":
        reducer = UMAP(n_neighbors=perplexity, n_components=2, transform_seed=SEED)
        reducer.fit(X=embs)
        embs = reducer.transform(X=embs)
    elif method == "t-SNE":
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=SEED)
        embs = tsne.fit_transform(X=embs)

    plt.rc(group="font", family="serif")
    plt.rc(group="text", usetex=True)

    classes = np.unique(ar=gt)
    # colors = plt.cm.tab10.colors[:len(classes)]
    labels = [
        r"$\cos x$", r"$\cos(-x)$", r"$\sin(x)/\tan(x)$",
        r"$1/\sec(x)$", r"$\sin(\sin^{-1}(\cos(x)))$",
        r"$\ln(x)$",
        r"$\sin(x)$",
        r"$\cot(x)$",
        r"$\sin^{-1}(x)$",
        r"$\cos^{-1}(x)$",
        r"$\cosh(x)$",
        r"$\tanh(x)$",
        r"$\sinh^{-1}(x)$",
        r"$\tanh^{-1}(x)$",
    ]
    fontsize = 18
    x_offset = 1
    y_offset = 0

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    for cls in classes:
        if cls == 0:
            id = np.where(cls == gt)[0][:5]
        else:
            id = np.where(cls == gt)[0][0]
        ax.scatter(
            x=embs[id, 0],
            y=embs[id, 1],
            s=50,
            c=colors[cls],
            marker="*",
            alpha=0.75,
        )

    ax.text(
        x=embs[0, 0] + 6.5,
        y=embs[0, 1] - 1.475,
        s=labels[0],  # Label the expression
        fontsize=fontsize,
        ha='right',
        va='center',
        color=colors[0]  # Same color as the scatter point
    )
    ax.text(
        x=embs[1, 0] + 1.25,
        y=embs[1, 1] + 1.375 ,
        s=labels[1],  # Label the expression
        fontsize=fontsize,
        ha='left',
        va='center',
        color=colors[0]  # S ame color as the scatter point
    )
    ax.text(
        x=embs[2, 0] - 0.4,
        y=embs[2, 1] - 2.0,
        s=labels[2],  # Label the expression
        fontsize=fontsize,
        ha='right',
        va='center',
        color=colors[0]  # Same color as the scatter point
    )
    ax.text(
        x=embs[3, 0] - 0.5,
        y=embs[3, 1] + 2.25,
        s=labels[3],  # Label the expression
        fontsize=fontsize,
        ha='right', 
        va='center',
        color=colors[0]  # Same color as the scatter point
    )
    ax.text(
        x=embs[4, 0] + 0.25,
        y=embs[4, 1] + 4.0,
        s=labels[4],  # Label the expression
        fontsize=fontsize,
        ha='center',
        va='bottom',
        color=colors[0]  # Same color as the scatter point
    )
    id = np.where(np.array(1) == gt)[0][0]
    ax.text(
        x=embs[id, 0] - x_offset,
        y=embs[id, 1] + y_offset,
        s=labels[5],  # Label the expression
        fontsize=fontsize,
        ha='right',
        va='center',
        color=colors[1]  # Same color as the scatter point
    )
    id = np.where(np.array(2) == gt)[0][0]
    ax.text(
        x=embs[id, 0] - x_offset,
        y=embs[id, 1] + y_offset,
        s=labels[6],  # Label the expression
        fontsize=fontsize,
        ha='right',
        va='center',
        color=colors[2]  # Same color as the scatter point
    )
    id = np.where(np.array(3) == gt)[0][0]
    ax.text(
        x=embs[id, 0] - x_offset,
        y=embs[id, 1] + y_offset,
        s=labels[7],  # Label the expression
        fontsize=20,
        ha='right',
        va='center',
        color=colors[3]  # Same color as the scatter point
    )
    id = np.where(np.array(4) == gt)[0][0]
    ax.text(
        x=embs[id, 0] + x_offset + 0.5,
        y=embs[id, 1] + y_offset,
        s=labels[8],  # Label the expression
        fontsize=fontsize,
        ha='left',
        va='center',
        color=colors[4]  # Same color as the scatter point
    )
    id = np.where(np.array(5) == gt)[0][0]
    ax.text(
        x=embs[id, 0] - x_offset,
        y=embs[id, 1] + y_offset,
        s=labels[9],  # Label the expression
        fontsize=fontsize,
        ha='right',
        va='center',
        color=colors[5]  # Same color as the scatter point
    )
    id = np.where(np.array(6) == gt)[0][0]
    ax.text(
        x=embs[id, 0] - x_offset,
        y=embs[id, 1] + y_offset,
        s=labels[10],  # Label the expression
        fontsize=fontsize,
        ha='right',
        va='center',
        color=colors[6]  # Same color as the scatter point
    )
    id = np.where(np.array(7) == gt)[0][0]
    ax.text(
        x=embs[id, 0] - x_offset,
        y=embs[id, 1] + y_offset,
        s=labels[11],  # Label the expression
        fontsize=fontsize,
        ha='right',
        va='center',
        color=colors[7]  # Same color as the scatter point
    )
    id = np.where(np.array(8) == gt)[0][0]
    ax.text(
        x=embs[id, 0] - x_offset,
        y=embs[id, 1] + y_offset,
        s=labels[12],  # Label the expression
        fontsize=fontsize,
        ha='right',
        va='center',
        color=colors[8]  # Same color as the scatter point
    )
    id = np.where(np.array(9) == gt)[0][0]
    ax.text(
        x=embs[id, 0] - x_offset,
        y=embs[id, 1] + y_offset,
        s=labels[13],  # Label the expression
        fontsize=fontsize,
        ha='right',
        va='center',
        color=colors[9]  # Same color as the scatter point
    )

    ax.set_xlabel('Component 1', fontsize=18, fontweight=2)
    ax.set_ylabel('Component 2', fontsize=18, fontweight=2)
    ax.set_xticks(np.arange(-40.0, 30.0, step=10))
    ax.set_yticks(np.arange(-50.0, 30.0, step=15))
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.spines["top"].set_visible(b=False)
    ax.spines["bottom"].set_visible(b=True)
    ax.spines["left"].set_visible(b=True)
    ax.spines["right"].set_visible(b=False)

    plt.tight_layout(rect=[0, 0 , 1, 1])
    plt.savefig(f"{method}.svg", transparent=True, dpi=500, format="svg")
    # plt.savefig(f"{method}.png", transparent=False, dpi=100)

    return


def main() -> None:
    cfg = get_config(args=None)

    parser = argparse.ArgumentParser(
        prog="ir",
        description="ir plot"
    )
    parser.add_argument(
        "--ckpt_filepath",
        "-m",
        type=str,
        required=True,
        help="seq2seq checkpoint filepath",
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

    emb_plt(
        method=method,
        embs=embs,
        perplexity=cfg.DIM_RED.PERPLEXITY,
        gt=kmc.gt,
        exprs=kmc.exprs,
    )

    return


if __name__ == "__main__":
    main()
