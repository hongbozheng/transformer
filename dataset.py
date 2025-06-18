from torch import Tensor
from typing import Dict, List

import torch
from tokenizer import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class EquivExpr(Dataset):
    def __init__(self, filepath: str, tokenizer: Tokenizer, val: bool) -> None:
        super().__init__()
        self.exprs = []
        self.tokenizer = tokenizer
        self.val = val

        file = open(file=filepath, mode='r', encoding='utf-8')
        if not val:
            for line in file:
                expr_pair = line.strip().split(sep='\t')
                self.exprs.append((expr_pair[0], expr_pair[1]))
        else:
            for line in file:
                expr = line.strip()
                self.exprs.append(expr)
        file.close()

    def __len__(self) -> int:
        return len(self.exprs)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        expr = self.exprs[idx]

        if not self.val:
            src_token_ids = self.tokenizer.encode(expr=expr[0])
            tgt_token_ids = self.tokenizer.encode(expr=expr[1])
            return {"src_token_ids": src_token_ids, "tgt_token_ids": tgt_token_ids}
        else:
            src_token_ids = self.tokenizer.encode(expr=expr)
            return {"src_token_ids": src_token_ids}

    def collate_fn(self, batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        pad_id = self.tokenizer.sym2idx["PAD"]

        src_token_ids = [item['src_token_ids'] for item in batch]
        # [B, L]
        src_token_ids = pad_sequence(
            sequences=src_token_ids,
            batch_first=True,
            padding_value=pad_id,
        )
        # https://gmongaras.medium.com/how-do-self-attention-masks-work-72ed9382510f
        # [B, L]
        src_attn_mask = torch.eq(input=src_token_ids, other=pad_id) \
            .to(dtype=torch.bool)

        if not self.val:
            tgt_token_ids = [item['tgt_token_ids'] for item in batch]
            # [B, L]
            tgt_token_ids = pad_sequence(
                sequences=tgt_token_ids,
                batch_first=True,
                padding_value=pad_id,
            )
            batch, seq_len = tgt_token_ids.size()
            # don't need to feed last token, so -1
            # [B, L-1, L-1]
            tgt_attn_mask = torch.tril(
                input=torch.ones(size=(batch, seq_len - 1, seq_len - 1)),
                diagonal=0,
            ).to(dtype=torch.bool)
            # [B, L-1] -> [B, 1, L-1, L-1]
            tgt_pad_mask = torch.ne(input=tgt_token_ids[:, :-1], other=pad_id) \
                .to(dtype=torch.bool).unsqueeze(dim=1)
            # [B, L-1, L-1] & [B, L-1, L-1] -> [B, L-1, L-1]
            tgt_attn_mask &= tgt_pad_mask

            return {
                "src_token_ids": src_token_ids,
                "src_attn_mask": src_attn_mask,
                "tgt_token_ids": tgt_token_ids,
                "tgt_attn_mask": tgt_attn_mask,
            }

        return {
            "src_token_ids": src_token_ids,
            "src_attn_mask": src_attn_mask,
        }
