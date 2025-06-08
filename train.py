import logger
import os
import torch
import torch.nn as nn
import torch.optim as optim
from avg_meter import AverageMeter
from logger import timestamp
from timm.scheduler.scheduler import Scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import train_params


def train_epoch(
        model: nn.Module,
        ckpt_last: str,
        optimizer: optim.Optimizer,
        lr_scheduler: Scheduler,
        criterion: nn.Module,
        max_norm: float,
        device: torch.device,
        dataloader: DataLoader,
        epoch: int,
        init_batch: int,
        save_every_n_iters: int,
) -> float:
    loader_tqdm = tqdm(iterable=dataloader, position=1, leave=False)
    loader_tqdm.set_description(desc=f"[{timestamp()}] [Batch 0]", refresh=True)

    n_iters = len(dataloader)
    loss_meter = AverageMeter()

    for i, batch in enumerate(iterable=loader_tqdm):
        if i < init_batch:
            continue

        src = batch["src"].to(device=device)
        src_mask = batch["src_mask"].to(device=device)

        optimizer.zero_grad()
        embs = model.encode(x=src, mask=src_mask)

        # ---------- mean pool ----------
        src_mask = src_mask.squeeze(dim=(-3, -2))
        n_pad = src_mask.int().sum(dim=-1)
        eoe_ids = src_mask.size(dim=-1) - n_pad - 1
        batch_ids = torch.arange(
            start=0, end=src_mask.size(dim=0), dtype=torch.int64, device=src_mask.device
        )
        src_mask[batch_ids, eoe_ids] = True
        src_mask[:, 0] = True

        embs[src_mask] = 0.0
        embs = embs.mean(dim=-2, keepdim=False)
        embs = embs.view(-1, 3, embs.size(dim=-1))

        query = embs[:, 0, :]
        pos_key = embs[:, 1, :]
        neg_key = embs[:, 2:, :]

        loss = criterion(
            query=query,
            pos_key=pos_key,
            neg_key=neg_key,
        )
        # -------------------------------

        # ----------- max sim -----------
        # src_mask = src_mask.squeeze(dim=(-3, -2))
        # n_pad = src_mask.int().sum(dim=-1)
        # eoe_ids = src_mask.size(dim=-1) - n_pad - 1
        # batch_ids = torch.arange(
        #     start=0, end=src_mask.size(dim=0), dtype=torch.int64, device=src_mask.device
        # )
        # src_mask[batch_ids, eoe_ids] = True
        # src_mask[:, 0] = True

        # embs = embs.view(-1, 5, embs.size(dim=-2), embs.size(dim=-1))
        # src_mask = src_mask.view(-1, 5, src_mask.size(dim=-1))

        # query = embs[:, 0, :, :]
        # pos_key = embs[:, 1, :, :]
        # neg_key = embs[:, 2:, :, :]
        # query_mask = src_mask[:, 0]
        # pos_mask = src_mask[:, 1]
        # neg_mask = src_mask[:, 2:]

        # loss = criterion(
        #     query=query,
        #     query_mask=query_mask,
        #     pos_key=pos_key,
        #     pos_mask=pos_mask,
        #     neg_key=neg_key,
        #     neg_mask=neg_mask,
        # )
        # -------------------------------

        # ---------- 1st token ----------
        # embs = embs[:, 0, :]
        # embs = embs.view(-1, 5, embs.size(dim=-1))
        # query = embs[:, 0, :]
        # pos_key = embs[:, 1, :]
        # neg_key = embs[:, 2:, :]
        # loss = criterion(query=query, pos_key=pos_key, neg_key=neg_key)
        # -------------------------------

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        lr_scheduler.step_update(n_iters * epoch + i)

        loss_meter.update(loss.item(), n=src.size(dim=0))
        loader_tqdm.set_description(
            desc=f"[{timestamp()}] [Batch {i+1}] loss {loss_meter.avg:.6f}",
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
        dataloader: DataLoader,
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
    best_loss = float('inf')

    if os.path.exists(path=ckpt_last):
        ckpt = torch.load(f=ckpt_last, map_location=device, weights_only=False)
        model.load_state_dict(state_dict=ckpt["model_state_dict"])
        optimizer.load_state_dict(state_dict=ckpt["optimizer_state_dict"])
        lr_scheduler.load_state_dict(state_dict=ckpt["lr_scheduler_state_dict"])
        init_batch = ckpt["batch"]+1
        init_epoch = ckpt["epoch"]+1 if init_batch == 0 else ckpt["epoch"]
        filename = os.path.basename(p=ckpt_last)
        logger.log_info(f"Loaded `{filename}`")

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
            dataloader=dataloader,
            epoch=epoch,
            init_batch=init_batch,
            save_every_n_iters=save_every_n_iters,
        )

        init_batch = 0

        epoch_tqdm.write(
            s=f"[{timestamp()}] [Epoch {epoch}] loss {loss:.6f}"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "epoch": epoch,
                "batch": -1,
                "loss": loss,
            },
            ckpt_last,
        )
        epoch_tqdm.write(
            s=f"[{timestamp()}] [Epoch {epoch}] Saved last model to "
                f"`{ckpt_last}`"
        )
