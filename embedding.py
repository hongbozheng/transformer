#!/usr/bin/env python3


import argparse
import config
import logger
import os
import torch
import tqdm
from expemb import ExpEmbTx, Tokenizer


def embedding(model_path: str, filepath: str) -> None:
    logger.log_info("Loading model... ")
    tokenizer = torch.load(f=model_path)["tokenizer"]
    model = ExpEmbTx.load_from_checkpoint(model_path, tokenizer=tokenizer)
    logger.log_info("Finish loading model.")

    device = torch.device(device="cuda") if torch.cuda.is_available() else torch.device(device="cpu")
    model.to(device=device)

    logger.log_info("Loading data...")
    exprs = []
    file = open(file=filepath, mode='r')
    for line in file:
        exprs.append(line.strip())
    logger.log_info("Finish loading data.")

    logger.log_info("Generating expressions embedding...")
    embeds = []
    progbar = tqdm.tqdm(iterable=exprs)
    for expr in progbar:
        progbar.set_description(desc=f"[INFO]: Processing '{expr}'", refresh=True)
        embed = model.get_embedding(exp_list=[expr], mode="max").detach().cpu()
        embeds.append(embed)
    embeds = torch.cat(tensors=embeds, dim=0)

    # assert embeds.shape[0] == len(exprs)

    torch.save(obj=embeds, f="embeds.pt")

    logger.log_info("Finish generating expressions embedding, the expressions embedding are stored in 'embeds.pt' file.")

    return


def main() -> None:
    parser = argparse.ArgumentParser(prog="embedding", description="Create and store embeddings of expressions")
    parser.add_argument("--model_path", "-m", type=str, required=True, help="trained model path")
    parser.add_argument("--filepath", "-f", type=str, required=True, help="expressions filepath")

    args = parser.parse_args()
    model_path = args.model_path
    filepath = args.filepath

    embedding(model_path=model_path, filepath=filepath)

    return


if __name__ == "__main__":
    main()
