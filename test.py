from torch import Tensor

from config import START, END, N, TOL, SECS
import logger
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from avg_meter import AverageMeter
from logger import timestamp
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from val import equiv


def beam_search(
        model: nn.Module,
        device: torch.device,
        src: Tensor,
        src_mask: Tensor,
        beam_size: int,
        seq_len: int,
        tokenizer: Tokenizer,
) -> Tensor:
    # [batch, seq_len, emb_dim]
    memory = model.encode(x=src, mask=src_mask)
    print("src mask", src_mask.shape)
    batch_size, src_seq_len, emb_dim = memory.size()

    # [batch, 1] of "SOE"
    beam = torch.full(
        size=(batch_size, 1),
        fill_value=tokenizer.sym2idx["SOE"],
        dtype=torch.int64,
        device=device,
    )
    print("beam", beam.shape)
    print(beam)
    # beam_scores[:, 1:] = -1e9
    # beam_scores = beam_scores.view(-1)

    # [batch, beam_size, 1]
    done = torch.zeros(
        size=(batch_size, beam_size),
        dtype=torch.bool,
        device=device,
    )

    # expand the first top-k beam
    tgt_mask = torch.tril(
        input=torch.ones(
            size=(batch_size, 1, beam.size(dim=1), beam.size(dim=1))
        ),
        diagonal=0,
    ).to(dtype=torch.uint8).to(device=device)
    # [batch x curr_len (1) x emb_dim]
    logits = model.decode(
        x=beam,
        memory=memory,
        tgt_mask=tgt_mask,
        mem_mask=src_mask,
    )
    print("logits", logits.shape)
    # [batch x vocab_size]
    logits = model.proj(x=logits[:, -1])
    vocab_size = logits.size(dim=-1)
    print("logits", logits.shape)
    # [batch x vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)
    print("log_probs", log_probs.shape)
    print(log_probs)
    beam_scores, nxt_tokens = torch.topk(log_probs, beam_size, dim=1, largest=True, sorted=True)
    beam_scores = beam_scores.view(batch_size*beam_size)
    print("beam_scores", beam_scores.shape)
    print(beam_scores)
    print("nxt_tokens", nxt_tokens.shape)
    print(nxt_tokens)

    beam = beam.unsqueeze(dim=1).expand(-1, beam_size, -1)
    print("beam", beam.shape)
    print(beam)
    nxt_tokens = nxt_tokens.view(batch_size, beam_size, 1)
    print("nxt_tokens", nxt_tokens.shape)
    print(nxt_tokens)
    beam = torch.cat(tensors=[beam, nxt_tokens], dim=-1)
    print("beam", beam.shape)
    print(beam)
    print("=============over============================================")

    # [batch, src_seq_len, emb_dim] -> [batch, beam, src_seq_len, emb_dim]
    memory = memory.unsqueeze(dim=1).expand(-1, beam_size, -1, -1)
    # [batch, beam, src_seq_len, emb_dim] -> [batch*beam, src_seq_len, emb_dim]
    memory = memory.contiguous().view(batch_size*beam_size, src_seq_len, emb_dim)
    print("mem", memory.shape)
    # [batch, 1, 1, src_seq_len] -> [batch, beam, 1, 1, src_seq_len]
    src_mask = src_mask.unsqueeze(dim=1).expand(-1, beam_size, -1, -1, -1)
    # [batch, beam, 1, 1, src_seq_len] -> [batch*beam, 1, 1, src_seq_len]
    src_mask = src_mask.contiguous().view(batch_size*beam_size, 1, 1, src_seq_len)
    print(src_mask.shape)

    for i in range(seq_len-2):
        # [batch, beam, curr_len] -> [batch*beam, curr_len]
        beam = beam.view(batch_size*beam_size, -1)
        tgt_mask = torch.tril(
            input=torch.ones(
                size=(batch_size*beam_size, 1, beam.size(dim=1), beam.size(dim=1))
            ),
            diagonal=0,
        ).to(dtype=torch.uint8).to(device=device)
        print("tgt mask", tgt_mask.shape)
        # [batch*beam x curr_len x emb_dim]
        logits = model.decode(
            x=beam,
            memory=memory,
            tgt_mask=tgt_mask,
            mem_mask=src_mask,
        )
        print("decode", logits.shape)

        # [batch*beam x vocab_size]
        logits = model.proj(x=logits[:, -1])
        print("proj", logits.shape)

        # [batch*beam, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)
        print("log_probs", log_probs.shape)
        print(log_probs)
        # log_probs = log_probs.view(batch_size, beam_size, vocab_size)
        # print("log_probs", log_probs.shape)
        # print(log_probs)
        print("beam scores", beam_scores.unsqueeze(dim=-1).expand(-1, vocab_size).shape)
        print(beam_scores.unsqueeze(dim=-1).expand(-1, vocab_size))
        log_probs += beam_scores.unsqueeze(dim=-1).expand(-1, vocab_size)
        print("log_probs", log_probs.shape)
        print(log_probs)
        log_probs = log_probs.view(batch_size, beam_size*vocab_size)
        print("log_probs", log_probs.shape)
        print(log_probs)
        nxt_scores, nxt_tokens = torch.topk(log_probs, beam_size, dim=1, largest=True, sorted=True)
        print("nxt_scores", nxt_scores.shape)
        print(nxt_scores)
        print("nxt_tokens", nxt_tokens.shape)
        print(nxt_tokens)

        beam_ids = torch.div(nxt_tokens, vocab_size, rounding_mode="floor")
        print("beam id", beam_ids.shape)
        print(beam_ids)
        nxt_tokens = nxt_tokens % vocab_size
        print("nxt_tokens", nxt_tokens.shape)
        print(nxt_tokens)

        beam_ids = beam_ids.unsqueeze(dim=-1).expand(-1, -1, beam.size(dim=-1))
        print("beam ids", beam_ids.shape)
        print(beam_ids)

        beam = beam.view(batch_size, beam_size, -1)
        beam = beam.gather(dim=1, index=beam_ids)
        print("beam", beam.shape)
        print(beam)

        beam = torch.cat(tensors=[beam, nxt_tokens.unsqueeze(dim=-1)], dim=-1)
        print("beam", beam.shape)
        print(beam)

        # exit()

        # print("beam", beam.gather(dim=1, index=beam_idx.unsqueeze(dim=-1).repeat(1, 1, beam.size(dim=-1))))
        # print("token", tokens.unsqueeze(dim=-1))
        # exit()

        # new_beam = torch.cat(tensors=[beam, tokens.unsqueeze(dim=-1)], dim=-1)
        # print("new_beam", new_beam.shape)
        # print(new_beam)

        # scores = nxt_scores
        # beam = new_beam

        done |= (nxt_tokens == tokenizer.sym2idx["EOE"])
        print(done)

        if done.all():
            break

        # exit()

        # beam = torch.cat(tensors=[beam.gather(dim=1, index=beam_idx), tokens], dim=-1)

        # print(logits.squeeze().shape)
        # prob, nxt_words = logits.squeeze().log_softmax(dim=-1).topk(k=beam_size, axis=-1)
        # tgt = tgt.repeat((beam_size, 1))
        # print(tgt.shape)
        # nxt_words = nxt_words.reshape(-1, 1)
        # print(nxt_words.shape)

    return beam


def calc_acc(src: Tensor, tgt: Tensor, tokenizer: Tokenizer) -> float:
    corrects = 0
    tot = src.size(dim=0)

    for s, tt in zip(src, tgt):
        s = tokenizer.decode(tokens=s)
        s = " ".join(s.split(sep=" ")[1:-1])
        for t in tt:
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
                break

    return corrects / tot


def test_model(
        model: nn.Module,
        device: torch.device,
        ckpt_filepath: str,
        test_loader: DataLoader,
        seq_len: int,
        tokenizer: Tokenizer,
) -> None:
    model.to(device=device)
    model.eval()

    if os.path.exists(path=ckpt_filepath):
        ckpt = torch.load(f=ckpt_filepath, map_location=device)
        model.load_state_dict(state_dict=ckpt["model"])
        filename = os.path.basename(p=ckpt_filepath)
        logger.log_info(f"Loaded '{filename}'")

    loader_tqdm = tqdm(iterable=test_loader, position=0, leave=False)
    loader_tqdm.set_description(desc=f"[{timestamp()}] [Batch 0]", refresh=True)

    acc_meter = AverageMeter()

    with torch.no_grad():
        for i, batch in enumerate(loader_tqdm):
            src = batch["src"].to(device=device)
            src_mask = batch["src_mask"].to(device=device)
            print(src, src_mask.shape)
            # [batch x n_beams x seq_len]
            beam = beam_search(
                model=model,
                device=device,
                src=src,
                src_mask=src_mask,
                beam_size=10,
                seq_len=seq_len,
                tokenizer=tokenizer,
            )
            acc = calc_acc(src=src, tgt=beam, tokenizer=tokenizer)
            acc_meter.update(val=acc, n=src.size(dim=0))

            loader_tqdm.set_description(
                desc=f"[{timestamp()}] [Batch {i+1}]: "
                     f"test acc {acc_meter.avg:.6f}",
                refresh=True,
            )
            print(f"{acc_meter.avg:.6f}")
