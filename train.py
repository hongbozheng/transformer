from torch import Tensor
from typing import Optional

import os
import torch
import torch.nn as nn
import torch.optim as optim
from avg_meter import AverageMeter
from logger import log_info, timestamp
from timm.scheduler.scheduler import Scheduler
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from val import val_epoch
from utils import train_params


def compute_loss(
        postprocess: Optional[str],
        criterion: nn.Module,
        logits: Tensor,
        gt: Optional[Tensor],
        src_attn_mask: Tensor,
        n_exprs: Optional[int],
) -> float:
    if postprocess is None:
        return criterion(
            input=logits.view(-1, logits.size(dim=-1)),
            target=gt.reshape(-1),
        )

    if postprocess == "cls":
        logits = logits[:, 0, :].view(-1, n_exprs, logits.size(dim=-1))

    elif postprocess in {"mean", "max", "maxsim"}:
        eoe_ids = src_attn_mask.int().sum(dim=-1) - 1
        batch_ids = torch.arange(
            start=0,
            end=src_attn_mask.size(dim=0),
            dtype=torch.int64,
            device=src_attn_mask.device,
        )
        src_attn_mask[batch_ids, eoe_ids] = False
        src_attn_mask[:, 0] = False

        if postprocess == "mean":
            logits[~src_attn_mask] = 0.0
            logits = logits.sum(dim=-2, keepdim=False)
            n_tokens = src_attn_mask.int().sum(dim=1, keepdim=False) \
                .float().unsqueeze(dim=-1)
            logits = logits / n_tokens
        elif postprocess == "max":
            logits[~src_attn_mask] = float("-inf")
            logits = logits.max(dim=-2, keepdim=False).values

        elif postprocess == "maxsim":
            _, L, D = logits.size()
            logits = logits.view(-1, n_exprs, L, D)
            src_attn_mask = src_attn_mask.view(-1, n_exprs, L)
            query = logits[:, 0, :, :]
            pos_key = logits[:, 1, :, :]
            neg_key = logits[:, 2:, :, :]
            query_mask = src_attn_mask[:, 0, :]
            pos_mask = src_attn_mask[:, 1, :]
            neg_mask = src_attn_mask[:, 2:, :]

            return criterion(
                query=query,
                pos_key=pos_key,
                neg_key=neg_key,
                query_mask=query_mask,
                pos_mask=pos_mask,
                neg_mask=neg_mask,
            )

        logits = logits.view(-1, n_exprs, logits.size(dim=-1))
        query = logits[:, 0, :]
        pos_key = logits[:, 1, :]
        neg_key = logits[:, 2:, :]

        return criterion(query=query, pos_key=pos_key, neg_key=neg_key)


def train_epoch(
        model: nn.Module,
        ckpt_last: str,
        optimizer: optim.Optimizer,
        lr_scheduler: Scheduler,
        postprocess: Optional[str],
        n_exprs: Optional[str],
        criterion: nn.CrossEntropyLoss,
        max_norm: float,
        device: torch.device,
        train_loader: DataLoader,
        epoch: int,
        init_batch: int,
        save_every_n_iters: int,
) -> float:
    model.train(mode=True)

    loader_tqdm = tqdm(iterable=train_loader, position=1, leave=False)
    loader_tqdm.set_description(desc=f"[{timestamp()}] [Batch 0]", refresh=True)

    n_iters = len(train_loader)
    loss_meter = AverageMeter()

    for i, batch in enumerate(iterable=loader_tqdm):
        if i < init_batch:
            continue

        src_token_ids = batch["src_token_ids"].to(device=device)
        src_attn_mask = batch["src_attn_mask"].to(device=device)
        if postprocess is None:
            tgt_token_ids = batch["tgt_token_ids"].to(device=device)
            tgt_attn_mask = batch["tgt_attn_mask"].to(device=device)
            gt = tgt_token_ids[:, 1:]
            tgt_token_ids = tgt_token_ids[:, :-1]
        else:
            tgt_token_ids = None
            tgt_attn_mask = None
            gt = None

        optimizer.zero_grad()
        logits = model(
            src_token_ids=src_token_ids,
            src_attn_mask=src_attn_mask,
            tgt_token_ids=tgt_token_ids,
            tgt_attn_mask=tgt_attn_mask,
        )
        loss = compute_loss(
            postprocess=postprocess,
            criterion=criterion,
            logits=logits,
            gt=gt,
            src_attn_mask=src_attn_mask,
            n_exprs=n_exprs,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        lr_scheduler.step_update(n_iters * epoch + i)

        loss_meter.update(loss.item(), n=src_token_ids.size(dim=0))
        loader_tqdm.set_description(
            desc=f"[{timestamp()}] [Batch {i+1}]: "
                 f"train loss {loss_meter.avg:.6f}",
            refresh=True,
        )

        if (i + 1) % save_every_n_iters == 0:
            n_steps = n_iters * epoch + (i+1)
            for param_group in optimizer.param_groups:
                loader_tqdm.write(f"[{timestamp()}] [Step {n_steps}] Current LR {param_group['lr']:.8f}")

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "batch": i,
                    "loss": loss,
                },
                ckpt_last,
            )
            loader_tqdm.write(
                s=f"[{timestamp()}] [Epoch {epoch}] [Batch {i}] Saved model to "
                  f"`{ckpt_last}`"
            )

    return loss_meter.avg


def train_model(
        model: nn.Module,
        ckpt_best: str,
        ckpt_last: str,
        optimizer: optim.Optimizer,
        lr_scheduler: Scheduler,
        postprocess: Optional[str],
        n_exprs: Optional[str],
        criterion: nn.CrossEntropyLoss,
        max_norm: float,
        device: torch.device,
        n_epochs: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        seq_len: int,
        tokenizer: Tokenizer,
        save_every_n_iters: int,
) -> None:
    path, _ = os.path.split(p=ckpt_last)
    if not os.path.exists(path=path):
        os.makedirs(name=path, exist_ok=True)

    model.to(device=device)

    params = train_params(model=model)
    log_info(f"Total trainable parameters {params * 1e-6:4f}M")

    init_epoch = 0
    init_batch = 0
    best_acc = 0.0

    if os.path.exists(path=ckpt_last):
        ckpt = torch.load(f=ckpt_last, map_location=device, weights_only=False)
        model.load_state_dict(state_dict=ckpt["model_state_dict"])
        optimizer.load_state_dict(state_dict=ckpt["optimizer_state_dict"])
        lr_scheduler.load_state_dict(state_dict=ckpt["lr_scheduler_state_dict"])
        init_batch = ckpt["batch"]+1
        init_epoch = ckpt["epoch"]+1 if init_batch == 0 else ckpt["epoch"]
        best_acc = ckpt["best_acc"]
        filename = os.path.basename(p=ckpt_last)
        log_info(f"Loaded `{filename}`")

    epoch_tqdm = tqdm(
        iterable=range(init_epoch, n_epochs),
        desc=f"[{timestamp()}] [Epoch {init_epoch}]",
        position=0,
        leave=True,
    )

    for epoch in epoch_tqdm:
        epoch_tqdm.set_description(
            desc=f"[{timestamp()}] [Epoch {epoch}]",
            refresh=True,
        )
        loss = train_epoch(
            model=model,
            ckpt_last=ckpt_last,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            postprocess=postprocess,
            n_exprs=n_exprs,
            criterion=criterion,
            max_norm=max_norm,
            device=device,
            train_loader=train_loader,
            epoch=epoch,
            init_batch=init_batch,
            save_every_n_iters=save_every_n_iters,
        )
        if val_loader is not None:
            acc = val_epoch(
                model=model,
                val_loader=val_loader,
                device=device,
                seq_len=seq_len,
                tokenizer=tokenizer,
            )
            epoch_tqdm.write(s=f"[{timestamp()}] [Epoch {epoch}] loss {loss:.6f} acc {acc:.6f}")

            if acc > best_acc:
                best_acc = acc
                torch.save(
                    obj={
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "batch": -1,
                        "best_acc": best_acc,
                        "loss": loss,
                    },
                    f=ckpt_best,
                )
                epoch_tqdm.write(
                    s=f"[{timestamp()}] [Epoch {epoch}] Saved best model to "
                    f"`{ckpt_best}`"
                )
        else:
            epoch_tqdm.write(s=f"[{timestamp()}] [Epoch {epoch}] loss {loss:.6f}")

        init_batch = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "epoch": epoch,
                "batch": -1,
                "best_acc": best_acc,
                "loss": loss,
            },
            ckpt_last,
        )
        epoch_tqdm.write(
            s=f"[{timestamp()}] [Epoch {epoch}] Saved last model to "
                f"`{ckpt_last}`"
        )
