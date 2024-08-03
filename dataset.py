from typing import Dict, List

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tokenizer import Tokenizer


class EA(Dataset):
    def __init__(self, filepath: str, tokenizer: Tokenizer) -> None:
        super().__init__()
        self.exprs = []
        self.tokenizer = tokenizer

        file = open(file=filepath, mode='r', encoding="utf-8")
        for line in file:
            expr = line.strip()
            self.exprs.append(expr)
        file.close()

        return

    def __len__(self) -> int:
        return len(self.exprs)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        expr = self.exprs[idx]
        tokens = self.tokenizer.encode(expr=expr)
        return {"src": tokens}

    def collate_fn(self, batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        src = [item["src"] for item in batch]

        src = pad_sequence(
            sequences=src,
            batch_first=True,
            padding_value=self.tokenizer.comp2idx["PAD"],
        )
        # [batch_size, 1 (n_heads), 1, seq_len]
        src_mask = torch.ne(input=src, other=self.tokenizer.comp2idx["PAD"]) \
            .unsqueeze(dim=1).unsqueeze(dim=1).to(dtype=torch.uint8)

        return {
            "src": src,
            "src_mask": src_mask,
        }


class KMC(Dataset):
    def __init__(self, filepath: str, tokenizer: Tokenizer) -> None:
        super().__init__()
        self.exprs = []
        self.gt = []
        self.n_clusters = 0
        self.sizes = []
        self.tokenizer = tokenizer

        cls = 0
        size = 0

        file = open(file=filepath, mode='r', encoding="utf-8")
        for line in file:
            expr = line.strip()
            if expr:
                self.gt.append(cls)
                self.exprs.append(expr)
                size += 1
            else:
                cls += 1
                self.n_clusters += 1
                self.sizes.append(size)
                size = 0
        file.close()

        assert self.n_clusters == len(self.sizes)

        return

    def __len__(self) -> int:
        return len(self.exprs)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        expr = self.exprs[idx]
        tokens = self.tokenizer.encode(expr=expr)
        return {"src": tokens}

    def collate_fn(self, batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        src = [item["src"] for item in batch]

        src = pad_sequence(
            sequences=src,
            batch_first=True,
            padding_value=self.tokenizer.comp2idx["PAD"],
        )
        # [batch_size, 1 (n_heads), 1, seq_len]
        src_mask = torch.ne(input=src, other=self.tokenizer.comp2idx["PAD"]) \
            .unsqueeze(dim=1).unsqueeze(dim=1).to(dtype=torch.uint8)

        return {
            "src": src,
            "src_mask": src_mask,
        }


class IR(Dataset):
    def __init__(self, filepath: str, tokenizer: Tokenizer) -> None:
        super().__init__()
        self.exprs = []
        self.gt = []
        self.tokenizer = tokenizer

        file = open(file=filepath, mode='r', encoding="utf-8")
        for i, line in enumerate(iterable=file):
            expr = line.strip()
            if expr:
                if i % 7 == 0:
                    expr, id = expr.split(sep='\t')
                    self.gt.append(int(id))
                self.exprs.append(expr)
        file.close()

        assert len(self.exprs) % 6 == 0
        assert len(self.gt) == len(self.exprs) / 6

        return

    def __len__(self) -> int:
        return len(self.exprs)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        expr = self.exprs[idx]
        tokens = self.tokenizer.encode(expr=expr)
        return {"src": tokens}

    def collate_fn(self, batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        src = [item["src"] for item in batch]

        src = pad_sequence(
            sequences=src,
            batch_first=True,
            padding_value=self.tokenizer.comp2idx["PAD"],
        )
        # [batch_size, 1 (n_heads), 1, seq_len]
        src_mask = torch.ne(input=src, other=self.tokenizer.comp2idx["PAD"]) \
            .unsqueeze(dim=1).unsqueeze(dim=1).to(dtype=torch.uint8)

        return {
            "src": src,
            "src_mask": src_mask,
        }
