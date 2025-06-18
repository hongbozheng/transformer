from torch import Tensor
from typing import Dict, List

import torch
from .registry import register_dataset
from tokenizer import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class ContrastiveExpr(Dataset):
    def __init__(self, filepath: str, tokenizer: Tokenizer) -> None:
        super().__init__()
        self.exprs = []
        self.tokenizer = tokenizer

        file = open(file=filepath, mode='r', encoding='utf-8')
        for line in file:
            expr = line.strip().split(sep='\t')
            self.exprs.append(expr)
        file.close()

    def __len__(self) -> int:
        return len(self.exprs)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        exprs = self.exprs[idx]
        token_ids = [self.tokenizer.encode(expr=expr) for expr in exprs]

        return {"token_ids": token_ids}

    def collate_fn(self, batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        pad_id = self.tokenizer.sym2idx["PAD"]

        token_ids = [ids for item in batch for ids in item["token_ids"]]
        # [B, L]
        token_ids = pad_sequence(
            sequences=token_ids,
            batch_first=True,
            padding_value=pad_id,
        )
        # https://gmongaras.medium.com/how-do-self-attention-masks-work-72ed9382510f
        # [B, L]
        attn_mask = torch.ne(input=token_ids, other=pad_id).to(dtype=torch.bool)

        return {
            "src_token_ids": token_ids,
            "src_attn_mask": attn_mask,
        }


@register_dataset(name="contrastive_expr")
def build_dataset(cfg, tokenizer) -> Dataset:
    return {
        "train": ContrastiveExpr(
            filepath=cfg.DATA.CONTRASTIVE_EXPR,
            tokenizer=tokenizer,
        ),
        "val": None
    }
