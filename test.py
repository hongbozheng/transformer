from torch import Tensor
from config import START, END, N, TOL, SECS
import torch
import torch.nn as nn
import torch.nn.functional as F
from logger import timestamp
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from avg_meter import AverageMeter
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
    batch_size = src.size(dim=0)

    # [batch_size, seq_len, emb_dim]
    memory = model.encode(x=src, mask=src_mask)
    print("src mask", src_mask.shape)
    # [batch_size, beam_size, seq_len, emb_dim]
    memory = memory.unsqueeze(dim=1).expand(batch_size, beam_size, -1, -1)
    # [batch_size*beam_size, seq_len, emb_dim]
    memory = memory.contiguous().view(batch_size*beam_size, -1, memory.size(dim=-1))
    print("mem", memory.shape)

    # [batch_size, 1] of "SOE"
    beam = torch.full(
        size=(batch_size, beam_size, 1),
        fill_value=tokenizer.comp2idx["SOE"],
        dtype=torch.int64,
        device=device,
    )
    scores = torch.zeros(size=(batch_size, beam_size), device=device)

    # [batch_size, 1]
    done = torch.zeros(
        size=(batch_size, beam_size),
        dtype=torch.bool,
        device=device
    )

    src_mask = src_mask.unsqueeze(dim=1).expand(batch_size, beam_size, -1, -1, -1)
    src_mask = src_mask.contiguous().view(batch_size*beam_size, -1, src_mask.size(dim=-2), src_mask.size(dim=-1))
    print(src_mask.shape)

    for i in range(seq_len-1):
        curr_beam = beam.view(batch_size*beam_size, -1)
        tgt_mask = torch.tril(
            input=torch.ones(
                size=(curr_beam.size(dim=0), 1, curr_beam.size(dim=1), curr_beam.size(dim=1))
            ),
            diagonal=0,
        ).to(dtype=torch.uint8).to(device=device)
        print("tgt mask", tgt_mask.shape)
        # [batch_size*beam_size x curr_len x emb_dim]
        logits = model.decode(
            x=curr_beam,
            memory=memory,
            tgt_mask=tgt_mask,
            mem_mask=src_mask,
        )
        print("decode", logits.shape)
        # [batch_size*beam_size x vocab_size]
        logits = model.proj(x=logits[:, -1])
        print("proj", logits.shape)
        log_probs = F.log_softmax(logits, dim=-1)
        print("log_probs", log_probs.shape)
        log_probs = log_probs.view(batch_size, beam_size, -1)
        print("log_probs", log_probs.shape)
        print(log_probs)
        new_scores = scores.unsqueeze(dim=-1) + log_probs
        print("new_scores", new_scores.shape)
        new_scores = new_scores.view(batch_size, -1)
        print("new_scores", new_scores.shape)
        top_scores, top_idx = new_scores.topk(beam_size, dim=-1, largest=True, sorted=True)
        print("top_scores", top_scores.shape)
        print(top_scores)
        print("top_idx", top_idx.shape)
        print(top_idx)
        beam_idx = top_idx // log_probs.size(dim=-1)
        print("beam idx", beam_idx.shape)
        print(beam_idx)
        tokens = top_idx % log_probs.size(dim=-1)
        print("tokens", tokens.shape)
        print(tokens)

        new_beam = torch.cat(tensors=[beam.gather(dim=1, index=beam_idx.unsqueeze(dim=-1).repeat(1, 1, beam.size(dim=-1))), tokens.unsqueeze(dim=-1)], dim=-1)
        print("new_beam", new_beam.shape)

        scores = top_scores
        beam = new_beam

        done |= (beam[:, :, -1] == tokenizer.comp2idx["EOE"])
        print(done)

        if done.all():
            break

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
        test_loader: DataLoader,
        device: torch.device,
        seq_len: int,
        tokenizer: Tokenizer,
):
    model.eval()

    loader_tqdm = tqdm(iterable=test_loader, position=0, leave=False)
    loader_tqdm.set_description(desc=f"[{timestamp()}] [Batch 0]", refresh=True)

    acc_meter = AverageMeter()

    with torch.no_grad():
        for i, batch in enumerate(loader_tqdm):
            src = batch["src"].to(device=device)
            src_mask = batch["src_mask"].to(device=device)
            print(src, src_mask.shape)
            # [batch x beam_size]
            beam = beam_search(
                model=model,
                device=device,
                src=src,
                src_mask=src_mask,
                beam_size=5,
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

    return
