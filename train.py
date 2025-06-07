import logger
import os
import torch
import torch.nn as nn
import torch.optim as optim
from avg_meter import AverageMeter
from logger import timestamp
from timm.scheduler.scheduler import Scheduler
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from val import val_epoch
from utils import train_params


def train_epoch(
        model: nn.Module,
        ckpt_last: str,
        optimizer: optim.Optimizer,
        lr_scheduler: Scheduler,
        criterion: nn.CrossEntropyLoss,
        max_norm: float,
        device: torch.device,
        train_loader: DataLoader,
        epoch: int,
        init_batch: int,
        save_every_n_iters: int,
) -> float:
    loader_tqdm = tqdm(iterable=train_loader, position=1, leave=False)
    loader_tqdm.set_description(desc=f"[{timestamp()}] [Batch 0]", refresh=True)

    n_iters = len(train_loader)
    loss_meter = AverageMeter()

    for i, batch in enumerate(iterable=loader_tqdm):
        if i < init_batch:
            continue

        src = batch["src"].to(device=device)
        tgt = batch["tgt"].to(device=device)
        src_mask = batch["src_mask"].to(device=device)
        tgt_mask = batch["tgt_mask"].to(device=device)

        tgt_input = tgt[:, :-1]

        optimizer.zero_grad()
        logits = model(
            src=src,
            tgt=tgt_input,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        tgt_output = tgt[:, 1:]
        loss = criterion(
            input=logits.reshape(-1, logits.size(dim=-1)),
            target=tgt_output.reshape(-1)
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        lr_scheduler.step_update(n_iters * epoch + i)

        loss_meter.update(loss.item(), n=src.size(dim=0))
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
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "lr_scheduler_state": lr_scheduler.state_dict(),
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
        criterion: nn.CrossEntropyLoss,
        max_norm: float,
        device: torch.device,
        n_epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        seq_len: int,
        tokenizer: Tokenizer,
        save_every_n_iters: int,
) -> None:
    path, _ = os.path.split(p=ckpt_last)
    if not os.path.exists(path=path):
        os.makedirs(name=path, exist_ok=True)

    model.to(device=device)
    model.train(mode=True)

    params = train_params(model=model)
    logger.log_info(f"Total trainable parameters {params * 1e-6}M")

    init_epoch = 0
    init_batch = 0
    best_acc = 0.0
    avg_losses = []  # TODO: Store train losses & val acc in JSON?

    if os.path.exists(path=ckpt_last):
        ckpt = torch.load(f=ckpt_last, map_location=device)
        model.load_state_dict(state_dict=ckpt["model_state_dict"])
        optimizer.load_state_dict(state_dict=ckpt["optimizer_state_dict"])
        lr_scheduler.load_state_dict(state_dict=ckpt["lr_scheduler_state_dict"])
        init_batch = ckpt["batch"]+1
        init_epoch = ckpt["epoch"]+1 if init_batch == 0 else ckpt["epoch"]
        best_acc = ckpt["best_acc"]
        filename = os.path.basename(p=ckpt_last)
        logger.log_info(f"Loaded `{filename}`.")

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
            criterion=criterion,
            max_norm=max_norm,
            device=device,
            train_loader=train_loader,
            epoch=epoch,
            init_batch=init_batch,
            save_every_n_iters=save_every_n_iters,
        )
        acc = val_epoch(
            model=model,
            val_loader=val_loader,
            device=device,
            seq_len=seq_len,
            tokenizer=tokenizer,
        )

        epoch_tqdm.write(
            s=f"[{timestamp()}] [Epoch {epoch}] loss {loss:.6f} acc {acc:.6f}"
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(
                obj={
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
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

        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "lr_scheduler_state": lr_scheduler.state_dict(),
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
