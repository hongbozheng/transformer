from typing import List, Dict
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tokenizer import Tokenizer


def causal_mask(size):
    # mask = torch.tril(torch.ones((1, size, size)), diagonal=0).type(torch.int)
    # return mask == 0
    mask = (torch.triu(torch.ones((size, size))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
        mask == 1, float(0.0))
    return mask


class EquivExpr(Dataset):
    def __init__(self, filepath: str, tokenizer: Tokenizer) -> None:
        super().__init__()
        self.expr_pairs = []
        self.tokenizer = tokenizer

        file = open(file=filepath, mode='r', encoding='utf-8')
        for line in file:
            expr_pair = line.strip().split(sep='\t')
            self.expr_pairs.append((expr_pair[0], expr_pair[1]))
        file.close()
        return

    def __len__(self) -> int:
        return len(self.expr_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        expr_pair = self.expr_pairs[idx]
        src_tokens = self.tokenizer.encode(expr=expr_pair[0])
        tgt_tokens = self.tokenizer.encode(expr=expr_pair[1])

        return {"src": src_tokens, "tgt": tgt_tokens}

    def collate_fn(self, batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        src = [item['src'] for item in batch]
        tgt = [item['tgt'] for item in batch]
        src = pad_sequence(
            sequences=src,
            batch_first=True,
            padding_value=self.tokenizer.comp2idx["PAD"],
        )
        tgt = pad_sequence(
            sequences=tgt,
            batch_first=True,
            padding_value=self.tokenizer.comp2idx["PAD"],
        )
        print(tgt)
        print(src)
        print("src_mask")
        # [batch_size, n_heads, 1, seq_len]
        src_mask = torch.ne(input=src, other=self.tokenizer.comp2idx["PAD"])\
            .unsqueeze(dim=1).unsqueeze(dim=1).to(dtype=torch.uint8)
        print(src_mask, src_mask.size())
        tgt_mask = torch.tril(
            input=torch.ones(
                size=(tgt.size(dim=0), 1, tgt.size(dim=1)-1, tgt.size(dim=1)-1)
            ),
            diagonal=0,
        ).to(dtype=torch.uint8)
        print(tgt_mask, tgt_mask.size())
        tgt_pad_mask = torch.ne(
            input=tgt[:, :-1],
            other=self.tokenizer.comp2idx["PAD"]
        ).unsqueeze(dim=1).unsqueeze(dim=1).to(dtype=torch.uint8)
        print("tgt_pad_mask")
        print(tgt_pad_mask, tgt_pad_mask.size())
        tgt_mask &= tgt_pad_mask
        print(tgt_mask, tgt_mask.size())

        return {
            "src": src,
            "tgt": tgt,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask,
        }