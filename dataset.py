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
        return

    def __len__(self) -> int:
        return len(self.exprs)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        expr = self.exprs[idx]

        if not self.val:
            src_tokens = self.tokenizer.encode(expr=expr[0])
            tgt_tokens = self.tokenizer.encode(expr=expr[1])
            return {"src": src_tokens, "tgt": tgt_tokens}
        else:
            src_tokens = self.tokenizer.encode(expr=expr)
            return {"src": src_tokens}

    def collate_fn(self, batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        src = [item['src'] for item in batch]
        src = pad_sequence(
            sequences=src,
            batch_first=True,
            padding_value=self.tokenizer.comp2idx["PAD"],
        )
        # https://gmongaras.medium.com/how-do-self-attention-masks-work-72ed9382510f
        # [batch_size, n_heads, 1, seq_len]
        src_mask = torch.eq(input=src, other=self.tokenizer.comp2idx["PAD"]) \
            .unsqueeze(dim=1).unsqueeze(dim=1).to(dtype=torch.bool)

        if not self.val:
            tgt = [item['tgt'] for item in batch]
            tgt = pad_sequence(
                sequences=tgt,
                batch_first=True,
                padding_value=self.tokenizer.comp2idx["PAD"],
            )
            # don't need to feed last token, so -1
            tgt_mask = torch.triu(
                input=torch.ones(
                    size=(tgt.size(dim=0), 1, tgt.size(dim=1) - 1,
                          tgt.size(dim=1) - 1)
                ),
                diagonal=1,
            ).to(dtype=torch.bool)
            tgt_pad_mask = torch.eq(
                input=tgt[:, :-1],
                other=self.tokenizer.comp2idx["PAD"]
            ).unsqueeze(dim=1).unsqueeze(dim=1).to(dtype=torch.bool)
            tgt_mask |= tgt_pad_mask

            '''
            print(tgt)
            print(src)
            print("src_mask")
            print(src_mask, src_mask.size())
            print("tgt_mask")
            print(tgt_mask, tgt_mask.size())
            print("tgt_pad_mask")
            print(tgt_pad_mask, tgt_pad_mask.size())
            print(tgt_mask, tgt_mask.size())
            '''
            return {
                "src": src,
                "tgt": tgt,
                "src_mask": src_mask,
                "tgt_mask": tgt_mask,
            }

        return {
            "src": src,
            "src_mask": src_mask,
        }
