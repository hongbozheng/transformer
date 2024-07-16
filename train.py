from torch.utils.data import DataLoader
import logger
from logger import timestamp
import os
import torch
import torch.nn as nn
import torch.optim as optim
from avg_meter import AverageMeter
from tqdm import tqdm
from tokenizer import Tokenizer
from val import val_epoch


def train_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        optimizer: optim.Optimizer,
        criterion: nn.CrossEntropyLoss,
        max_norm: float,
) -> float:
    model.train(mode=True)

    loader_tqdm = tqdm(iterable=train_loader, position=1, leave=False)
    loader_tqdm.set_description(desc=f"[{timestamp()}] [Batch 0]", refresh=True)
    # print(optimizer.param_groups[0]["lr"])

    loss_meter = AverageMeter()

    for i, batch in enumerate(iterable=loader_tqdm):
        src = batch["src"].to(device=device)
        src_mask = batch["src_mask"].to(device=device)

        optimizer.zero_grad()
        embs = model(src=src, src_mask=src_mask)
        # print(embs)
        # print(embs.size())
        src_mask = src_mask.squeeze(dim=1).squeeze(dim=1)
        # print(src_mask)
        # print(src_mask.size())
        embs[src_mask==0] = float("-inf")
        # print(embs)
        # print(embs.size())
        embs, _ = embs.max(dim=1, keepdim=False)
        # print(embs)
        # print(embs.size())

        embs = embs.reshape(3, -1, embs.size(dim=-1))

        loss = criterion(e=embs[0], e_pos=embs[1], e_neg=embs[2])
        # print(loss)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        loss_meter.update(loss.item(), n=src.size(dim=0))

        loader_tqdm.set_description(
            desc=f"[{timestamp()}] [Batch {i+1}]: "
                 f"train loss {loss_meter.avg:.6f}",
            refresh=True,
        )

    return loss_meter.avg


def train_model(
        model: nn.Module,
        device: torch.device,
        ckpt_filepath: str,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.LRScheduler,
        n_epochs: int,
        criterion: nn.CrossEntropyLoss,
        max_norm: float,
        train_loader: DataLoader,
        val_loader: DataLoader,
        seq_len: int,
        tokenizer: Tokenizer,
):
    model.to(device=device)
    model.train(mode=True)

    path, _ = os.path.split(p=ckpt_filepath)
    if not os.path.exists(path=path):
        os.makedirs(name=path, exist_ok=True)

    init_epoch = 0
    best_loss = float('inf')
    avg_losses = []  # TODO: Store train losses & val acc in JSON?

    if os.path.exists(path=ckpt_filepath):
        ckpt = torch.load(f=ckpt_filepath, map_location=device)
        model.load_state_dict(state_dict=ckpt["model"])
        optimizer.load_state_dict(state_dict=ckpt["optimizer"])
        lr_scheduler.load_state_dict(state_dict=ckpt["lr_scheduler"])
        init_epoch = ckpt["epoch"]+1
        best_acc = ckpt["best_acc"]
        filename = os.path.basename(p=ckpt_filepath)
        logger.log_info(f"Loaded '{filename}'")

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
        avg_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            max_norm=max_norm,
        )
        # avg_acc = val_epoch(
        #     model=model,
        #     val_loader=val_loader,
        #     device=device,
        #     seq_len=seq_len,
        #     tokenizer=tokenizer,
        # )
        lr_scheduler.step()

        epoch_tqdm.write(
            s=f"[{timestamp()}] [Epoch {epoch}]: loss {avg_loss:.6f} "
              # f"acc {avg_acc:.6f}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                obj={
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                },
                f=ckpt_filepath,
            )
            epoch_tqdm.write(
                s=f"[{timestamp()}] [Epoch {epoch}]: Saved best model to "
                  f"'{ckpt_filepath}'"
            )

    return
