from typing import Tuple
from torch import Tensor
from config import START, END, N, TOL, SECS
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from avg_meter import AverageMeter
from tokenizer import Tokenizer
from tqdm import tqdm
from timeout import timeout
import sympy as sp
from sympy import Expr, Symbol
from refactor import prefix_to_sympy
import numpy as np
from components import VARIABLES
import logger
from logger import timestamp


def greedy_decode(
        model: nn.Module,
        device: torch.device,
        src: Tensor,
        src_mask: Tensor,
        seq_len: int,
        tokenizer: Tokenizer,
) -> Tensor:
    batch_size = src.size(dim=0)

    memory = model.encode(x=src, mask=src_mask)

    # [batch, 1] of "SOE"
    tgt = torch.full(
        size=(batch_size, 1),
        fill_value=tokenizer.comp2idx["SOE"],
        dtype=torch.int64,
        device=device,
    )
    # [batch_size, 1]
    done = torch.zeros(
        size=(batch_size, 1),
        dtype=torch.bool,
        device=device
    )

    for i in range(seq_len-1):
        tgt_mask = torch.tril(
            input=torch.ones(
                size=(batch_size, 1, tgt.size(dim=1), tgt.size(dim=1))
            ),
            diagonal=0,
        ).to(dtype=torch.uint8).to(device=device)
        #print(tgt_mask, tgt_mask.size())
        logits = model.decode(
            x=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            mem_mask=src_mask,
        )
        logits = model.proj(x=logits[:, -1])
        #print("after proj")
        #print(logits, logits.size())
        _, nxt_tokens = torch.max(input=logits, dim=1, keepdim=True)
        #print("nxt words")
        #print(nxt_words, nxt_words.size())
        tgt = torch.cat(tensors=[tgt, nxt_tokens], dim=1)
        #print("tgt")
        #print(tgt, tgt.size())

        done |= (nxt_tokens == tokenizer.comp2idx["EOE"])
        #print("done", done)

        if done.all():
            break

    return tgt


def equiv(
        expr_pair: Tuple[str, str],
        start: float,
        end: float,
        n: int,
        tol: float,
        secs: int,
) -> bool:
    """
    Args:
        expr_pair: expr pair
        start: domain start
        end: domain end
        n: # of test vals
        tol: tolerance
        secs: timeout secs
    """
    @timeout(secs=secs)
    def _simplify(expr: Expr) -> Expr:
        return sp.simplify(expr=expr)

    @timeout(secs=secs)
    def _equiv(
            x: Symbol,
            expr: Expr,
            start: float,
            end: float,
            n: int,
            tol: float,
    ) -> bool:
        rand_nums = np.random.uniform(low=start, high=end, size=n)
        for num in rand_nums:
            val = expr.subs(x, num).evalf()
            if abs(val) > tol:
                return False
        return True

    x = VARIABLES['x']

    try:
        src = prefix_to_sympy(expr=expr_pair[0])
        tgt = prefix_to_sympy(expr=expr_pair[1])
    except Exception as e:
        logger.log_error(
            f"prefix_to_sympy exception {e}; {expr_pair[0]} & {expr_pair[1]}"
        )
        return False
    try:
        src = _simplify(expr=src)
        tgt = _simplify(expr=tgt)
    except Exception as e:
        logger.log_error(
            f"simplify exception {e}; {expr_pair[0]} & {expr_pair[1]}"
        )
        return False

    expr = src - tgt

    if expr == 0:
        logger.log_debug(
            f"simplify  , equiv    ; {expr_pair[0]} & {expr_pair[1]}"
        )
        return True
    else:
        try:
            # TODO: SIMPLIFY THIS PART AFTER MAKING SURE IT WORKS
            res = _equiv(
                x=x,
                expr=expr,
                start=start,
                end=end,
                n=n,
                tol=tol,
            )
            logger.log_debug(
                f"subs_evalf, equiv    ; {expr_pair[0]} & {expr_pair[1]}"
            )
            if res:
                logger.log_debug(
                    f"subs_evalf, equiv    ; {expr_pair[0]} & {expr_pair[1]}"
                )
            else:
                logger.log_error(
                    f"subs_evalf, non-equiv; {expr_pair[0]} & {expr_pair[1]}"
                )
            return res
        except Exception as e:
            logger.log_error(
                f"_check_equiv exception {e}; {expr_pair[0]} & {expr_pair[1]}"
            )
            return False


def calc_acc(src: Tensor, tgt: Tensor, tokenizer: Tokenizer) -> float:
    corrects = 0
    tot = src.size(dim=0)

    for s, t in zip(src, tgt):
        s = tokenizer.decode(tokens=s)
        s = " ".join(s.split(sep=" ")[1:-1])
        t = tokenizer.decode(tokens=t)
        t = " ".join(t.split(sep=" ")[1:-1])

        if equiv(
            expr_pair=(s, t),
            start=START,
            end=END,
            n=N,
            tol=TOL,
            secs=SECS,
        ):
            corrects += 1

    return corrects / tot


def val_epoch(
        model: nn.Module,
        device: torch.device,
        val_loader: DataLoader,
        seq_len: int,
        tokenizer: Tokenizer,
):
    model.eval()

    loader_tqdm = tqdm(iterable=val_loader, position=1, leave=False)
    loader_tqdm.set_description(desc=f"[{timestamp()}] [Batch 0]", refresh=True)

    acc_meter = AverageMeter()

    with torch.no_grad():
        for i, batch in enumerate(iterable=loader_tqdm):
            src = batch["src"].to(device=device)
            # tgt_expr = batch["tgt"].to(device=device)
            src_mask = batch["src_mask"].to(device=device)

            preds = greedy_decode(
                model=model,
                device=device,
                src=src,
                src_mask=src_mask,
                seq_len=seq_len,
                tokenizer=tokenizer,
            )
            # print(preds)
            # print(preds.size())
            acc = calc_acc(src=src, tgt=preds, tokenizer=tokenizer)
            acc_meter.update(val=acc, n=src.size(dim=0))

            loader_tqdm.set_description(
                desc=f"[{timestamp()}] [Batch {i+1}]: "
                     f"val acc {acc_meter.avg:.6f}",
                refresh=True,
            )

    return acc_meter.avg
