from torch import Tensor
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from avg_meter import AverageMeter
from tokenizer import Tokenizer
from tqdm import tqdm


def greedy_decode(
        model: nn.Module,
        device: torch.device,
        src: Tensor,
        src_mask: Tensor,
        seq_len: int,
        tokenizer: Tokenizer,
) -> Tensor:
    memory = model(src=src, src_mask=src_mask)
    pred = torch.full(
        size=(1, 1),
        fill_value=tokenizer.components["SOE"],
        dtype=torch.int64,
        device=device,
    )
    for i in range(seq_len-1):
        tgt_mask = causal_mask(size=pred.size(dim=0)).type(torch.bool).to(device=device)
        out = model.decode(x=, memory=memory, tgt_mask=tgt_mask, src_mask=src_mask)
        prob = model.proj(x=out[:, -1])
        _, nxt_word = torch.max(input=prob, dim=1)
        nxt_word = nxt_word.item()

        pred = torch.cat(
            tensors=[
                pred,
                torch.full(
                    size=(1, 1),
                    fill_value=nxt_word,
                    dtype=torch.int64,
                    device=device
                ),
            ], dim=0
        )

        if nxt_word == tokenizer.components["EOE"]:
            break
    return pred


def val_epoch(
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        seq_len: int,
        tokenizer: Tokenizer,
):
    model.eval()

    loader_tqdm = tqdm(iterable=val_loader, position=1)
    loader_tqdm.set_description(desc=f"[Batch 0]", refresh=True)

    acc_meter = AverageMeter()

    with torch.no_grad():
        for i, batch in enumerate(iterable=loader_tqdm):
            src_expr = batch["src"].to(device=device)
            tgt_expr = batch["tgt"].to(device=device)

            pred = greedy_decode(
                model=model,
                device=device,
                src=src_expr,
                src_mask=batch["src_mask"], # TODO: HANDLE MASK
                seq_len=seq_len,
                tokenizer=tokenizer,
            )

            tokenizer.decode(tokens=pred)

    return


def val_model():
    return