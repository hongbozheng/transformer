from torch.utils.data import DataLoader
import logger
import os
import torch
import torch.nn as nn
import torch.optim as optim
from avg_meter import AverageMeter
from tqdm import tqdm

from torch.nn import Transformer
from torch import Tensor
import math
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        # outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
        #                         src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                None, None,
                                memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

torch.autograd.set_detect_anomaly(True)
def train_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        criterion: nn.CrossEntropyLoss,
        optimizer: optim.Optimizer,
) -> float:
    # model = Seq2SeqTransformer(num_decoder_layers=6, num_encoder_layers=6, emb_size=512, nhead=8, src_vocab_size=100, tgt_vocab_size=100, dim_feedforward=512, dropout=0.1)


    model.train(mode=True)

    loader_tqdm = tqdm(iterable=train_loader, position=0, leave=True)
    loader_tqdm.set_description(desc=f"[Batch 0]", refresh=True)

    loss_meter = AverageMeter()

    for i, batch in enumerate(iterable=loader_tqdm):
        src = batch["src"].to(device=device)
        tgt = batch["tgt"].to(device=device)
        src_mask = batch["src_mask"].to(device=device)
        tgt_mask = batch["tgt_mask"].to(device=device)

        tgt_input = tgt[:, :-1]

        optimizer.zero_grad()
        logits = model(src=src, tgt=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
        print(logits)
        print(logits.size())
        tgt_output = tgt[:, 1:]
        loss = criterion(input=logits.reshape(-1, logits.size(dim=-1)), target=tgt_output.reshape(-1))
        loss.backward()
        # TODO: NOT SURE IF WE NEED TO AVOID GRADIENT EXPLODING ISSUE
        # TODO: WITH clip_grad_norm_
        optimizer.step()
        loss_meter.update(loss.item(), n=src.size(dim=0))

        loader_tqdm.set_description(
            desc=f"[Batch {i+1}]: train loss {loss_meter.avg:.6f}",
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
       train_loader: DataLoader,
       val_loader: DataLoader,
):
    path, _ = os.path.split(p=ckpt_filepath)
    if not os.path.exists(path=path):
        os.makedirs(name=path, exist_ok=True)

    model.to(device=device)

    init_epoch = 0
    best_acc = 0.0
    avg_losses = []

    if os.path.exists(path=ckpt_filepath):
        ckpt = torch.load(f=ckpt_filepath, map_location=device)
        model.loadstate_dict(state_dict=ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

        init_epoch = ckpt["epoch"]+1
        best_acc = ckpt["best_acc"]
        filename = os.path.basename(p=ckpt_filepath)
        logger.log_info(f"Loaded '{filename}'")

    epoch_tqdm = tqdm(
        iterable=range(init_epoch, n_epochs),
        position=0,
        leave=True
    )

    for epoch in epoch_tqdm:
        epoch_tqdm.set_description(desc=f"[Epoch {epoch}]", refresh=True)
        avg_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
        )
        print(avg_loss)
        # TODO: Call val_epoch here

        # if avg_acc > best_acc:
        #     best_acc = avg_acc
        #     torch.save(
        #         obj={
        #             "model": model.state_dict(),
        #             "optimizer": optimizer.state_dict(),
        #             "lr_scheduler": lr_scheduler.state_dict(),
        #             "epoch": epoch,
        #             "best_acc": best_acc,
        #         },
        #         f=ckpt_filepath,
        #     )

    return