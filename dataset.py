from torch import Tensor
from typing import Dict, List

import torch
from tokenizer import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class CL(Dataset):
    def __init__(self, filepath: str, tokenizer: Tokenizer) -> None:
        super().__init__()
        self.exprs = []
        self.tokenizer = tokenizer

        file = open(file=filepath, mode='r', encoding='utf-8')
        for line in file:
            expr_triplet = line.strip().split(sep='\t')
            self.exprs.append(
                (expr_triplet[0], expr_triplet[1], expr_triplet[2])
            )
        file.close()
        return

    def __len__(self) -> int:
        return len(self.exprs)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        expr = self.exprs[idx]

        src_tokens = self.tokenizer.encode(expr=expr[0])
        pos_tokens = self.tokenizer.encode(expr=expr[1])
        neg_tokens = self.tokenizer.encode(expr=expr[2])
        return {"src": src_tokens, "pos": pos_tokens, "neg": neg_tokens}

    def collate_fn(self, batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        src = []
        pos = []
        neg = []
        for item in batch:
            src.append(item["src"])
            pos.append(item["pos"])
            neg.append(item["neg"])
        src.extend(pos)
        src.extend(neg)

        src = pad_sequence(
            sequences=src,
            batch_first=True,
            padding_value=self.tokenizer.comp2idx["PAD"],
        )
        # https://gmongaras.medium.com/how-do-self-attention-masks-work-72ed9382510f
        # [batch_size, 1 (n_heads), 1, seq_len]
        src_mask = torch.ne(input=src, other=self.tokenizer.comp2idx["PAD"]) \
            .unsqueeze(dim=1).unsqueeze(dim=1).to(dtype=torch.uint8)

        return {
            "src": src,
            "src_mask": src_mask,
        }
