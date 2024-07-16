from typing import List, Dict
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tokenizer import Tokenizer


class CL(Dataset):
    def __init__(self, filepath: str, tokenizer: Tokenizer, val: bool) -> None:
        super().__init__()
        self.exprs = []
        self.tokenizer = tokenizer
        self.val = val

        file = open(file=filepath, mode='r', encoding='utf-8')
        if not val:
            for line in file:
                expr_triplet = line.strip().split(sep='\t')
                self.exprs.append(
                    (expr_triplet[0], expr_triplet[1], expr_triplet[2])
                )
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
            pos_tokens = self.tokenizer.encode(expr=expr[1])
            neg_tokens = self.tokenizer.encode(expr=expr[2])
            return {"src": src_tokens, "pos": pos_tokens, "neg": neg_tokens}
        else:
            src_tokens = self.tokenizer.encode(expr=expr)
            return {"src": src_tokens}

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
        # [batch_size, n_heads, 1, seq_len]
        src_mask = torch.ne(input=src, other=self.tokenizer.comp2idx["PAD"]) \
            .unsqueeze(dim=1).unsqueeze(dim=1).to(dtype=torch.uint8)

        # if not self.val:
        #     tgt = [item['tgt'] for item in batch]
        #     tgt = pad_sequence(
        #         sequences=tgt,
        #         batch_first=True,
        #         padding_value=self.tokenizer.comp2idx["PAD"],
        #     )
        #     # don't need to feed last token, so -1
        #     tgt_mask = torch.tril(
        #         input=torch.ones(
        #             size=(tgt.size(dim=0), 1, tgt.size(dim=1) - 1,
        #                   tgt.size(dim=1) - 1)
        #         ),
        #         diagonal=0,
        #     ).to(dtype=torch.uint8)
        #     tgt_pad_mask = torch.ne(
        #         input=tgt[:, :-1],
        #         other=self.tokenizer.comp2idx["PAD"]
        #     ).unsqueeze(dim=1).unsqueeze(dim=1).to(dtype=torch.uint8)
        #     tgt_mask &= tgt_pad_mask
        #
        #     '''
        #     print(tgt)
        #     print(src)
        #     print("src_mask")
        #     print(src_mask, src_mask.size())
        #     print("tgt_mask")
        #     print(tgt_mask, tgt_mask.size())
        #     print("tgt_pad_mask")
        #     print(tgt_pad_mask, tgt_pad_mask.size())
        #     print(tgt_mask, tgt_mask.size())
        #     '''
        #     return {
        #         "src": src,
        #         "tgt": tgt,
        #         "src_mask": src_mask,
        #         "tgt_mask": tgt_mask,
        #     }

        return {
            "src": src,
            "src_mask": src_mask,
        }


class CL_KMeans(Dataset):
    def __init__(self, filepath: str, tokenizer: Tokenizer) -> None:
        super().__init__()
        self.exprs = []
        self.n_clusters = 0
        self.sizes = []
        self.tokenizer = tokenizer

        size = 0

        file = open(file=filepath, mode='r', encoding='utf-8')
        for i, line in enumerate(file):
            expr = line.strip()
            if expr:
                self.exprs.append(expr)
                size += 1
            else:
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
        # https://gmongaras.medium.com/how-do-self-attention-masks-work-72ed9382510f
        # [batch_size, n_heads, 1, seq_len]
        src_mask = torch.ne(input=src, other=self.tokenizer.comp2idx["PAD"]) \
            .unsqueeze(dim=1).unsqueeze(dim=1).to(dtype=torch.uint8)

        # if not self.val:
        #     tgt = [item['tgt'] for item in batch]
        #     tgt = pad_sequence(
        #         sequences=tgt,
        #         batch_first=True,
        #         padding_value=self.tokenizer.comp2idx["PAD"],
        #     )
        #     # don't need to feed last token, so -1
        #     tgt_mask = torch.tril(
        #         input=torch.ones(
        #             size=(tgt.size(dim=0), 1, tgt.size(dim=1) - 1,
        #                   tgt.size(dim=1) - 1)
        #         ),
        #         diagonal=0,
        #     ).to(dtype=torch.uint8)
        #     tgt_pad_mask = torch.ne(
        #         input=tgt[:, :-1],
        #         other=self.tokenizer.comp2idx["PAD"]
        #     ).unsqueeze(dim=1).unsqueeze(dim=1).to(dtype=torch.uint8)
        #     tgt_mask &= tgt_pad_mask
        #
        #     '''
        #     print(tgt)
        #     print(src)
        #     print("src_mask")
        #     print(src_mask, src_mask.size())
        #     print("tgt_mask")
        #     print(tgt_mask, tgt_mask.size())
        #     print("tgt_pad_mask")
        #     print(tgt_pad_mask, tgt_pad_mask.size())
        #     print(tgt_mask, tgt_mask.size())
        #     '''
        #     return {
        #         "src": src,
        #         "tgt": tgt,
        #         "src_mask": src_mask,
        #         "tgt_mask": tgt_mask,
        #     }

        return {
            "src": src,
            "src_mask": src_mask,
        }
