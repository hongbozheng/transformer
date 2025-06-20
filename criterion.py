from torch import Tensor
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    def __init__(self, temperature: float, reduction: str) -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
            self,
            query: Tensor,
            pos_key: Tensor,
            neg_key: Tensor,
    ) -> Tensor:
        # [B, D] -> [B, D]
        query = F.normalize(input=query, p=2.0, dim=-1, eps=1e-12)
        # [B, D] -> [B, D]
        pos_key = F.normalize(input=pos_key, p=2.0, dim=-1, eps=1e-12)
        # [B, NG, D] -> [B, NG, D]
        neg_key = F.normalize(input=neg_key, p=2.0, dim=-1, eps=1e-12)

        # [B, D] * [B, D] -> [B, 1]
        pos_logit = torch.sum(query*pos_key, dim=1, keepdim=True)

        # [B, D] -> [B, 1, D]
        query = query.unsqueeze(dim=1)
        # [B, 1, D] @ [B, D, NG] -> [B, 1, NG]
        neg_logit = query @ neg_key.transpose(dim0=-2, dim1=-1)
        # [B, 1, NG] -> [B, NG]
        neg_logit = neg_logit.squeeze(dim=1)

        # [B, 1] [B, NG] -> [B, 1 + NG]
        logits = torch.cat(tensors=[pos_logit, neg_logit], dim=1)
        labels = torch.zeros(
            logits.size(dim=0),
            dtype=torch.int64,
            device=query.device,
        )
        loss = F.cross_entropy(
            input=logits/self.temperature,
            target=labels,
            reduction=self.reduction,
        )

        return loss


class SimCSE(nn.Module):
    def __init__(self, temperature: float, reduction: str) -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
            self,
            query: Tensor,
            pos_key: Tensor,
            neg_key: Tensor,
    ) -> Tensor:
        pos_logit = F.cosine_similarity(x1=query, x2=pos_key, dim=1, eps=1e-8)
        neg_logit = F.cosine_similarity(x1=query, x2=neg_key, dim=1, eps=1e-8)

        numerator = torch.exp(pos_logit/self.temperature)
        denominator = torch.exp(pos_logit/self.temperature)\
            +torch.exp(neg_logit/self.temperature)

        if self.reduction == "mean":
            loss = -torch.log(numerator/denominator).mean()
        elif self.reduction == "sum":
            loss = -torch.log(numerator/denominator).sum()
        else:
            raise ValueError(
                "Invalid reduction. Please choose from {{'mean', 'sum'}}."
            )

        return loss
    

class MaxSim(nn.Module):
    def __init__(self, temperature: float, reduction: str) -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
            self,
            query: Tensor,
            pos_key: Tensor,
            neg_key: Tensor,
            query_mask: Tensor,
            pos_mask: Tensor,
            neg_mask: Tensor,
    ) -> Tensor:
        # [B, L, D] -> [B, L, D]
        query = F.normalize(input=query, p=2.0, dim=-1, eps=1e-12)
        # [B, L, D] -> [B, L, D]
        pos_key = F.normalize(input=pos_key, p=2.0, dim=-1, eps=1e-12)
        # [B, NG, L, D] -> [B, NG, L, D]
        neg_key = F.normalize(input=neg_key, p=2.0, dim=-1, eps=1e-12)

        # [B. L, D] @ [B, D, L] -> [B, L, L]
        pos_logit = query @ pos_key.transpose(dim0=-2, dim1=-1)
        # [B, L] -> [B, 1, L]
        pos_mask = pos_mask.unsqueeze(dim=1)
        pos_logit = pos_logit.masked_fill(mask=~pos_mask, value=float('-inf'))
        # [B, L, L] -> [B, L]
        pos_logit = torch.max(input=pos_logit, dim=-1).values
        # [B, L] * [B, L] -> [B, 1]
        pos_logit = (pos_logit * query_mask.to(dtype=pos_logit.dtype)) \
            .sum(dim=-1, keepdim=True)

        # [B, L, D] - > [B, 1, L, D]
        query = query.unsqueeze(dim=1)
        # [B, 1, L, D] @ [B, NG, D, L] -> [B, NG, L, L]
        neg_logit = query @ neg_key.transpose(dim0=-2, dim1=-1)
        # [B, L] -> [B, 1, L]
        neg_mask = neg_mask.unsqueeze(dim=1)
        neg_logit = neg_logit.masked_fill(mask=~pos_mask, value=float('-inf'))
        # [B, NG, L, L] -> [B, NG, L]
        neg_logit = torch.max(input=neg_logit, dim=-1).values
        # [B, L] -> [B, 1, L]
        query_mask = query_mask.unsqueeze(dim=1)
        # [B, NG, L] * [B, 1, L] -> [B, NG]
        neg_logit = (neg_logit * query_mask.to(dtype=neg_logit.dtype)) \
            .sum(dim=-1, keepdim=False)

        # [B, 1] [B, NG] -> [B, 1 + NG]
        logits = torch.cat(tensors=[pos_logit, neg_logit], dim=1)
        labels = torch.zeros(
            logits.size(dim=0),
            dtype=torch.int64,
            device=query.device,
        )
        loss = F.cross_entropy(
            input=logits/self.temperature,
            target=labels,
            reduction=self.reduction,
        )

        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float, reduction: str) -> None:
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
            self,
            query: Tensor,
            pos_key: Tensor,
            neg_key: Tensor,
    ) -> Tensor:
        pos_logit = F.cosine_similarity(x1=query, x2=pos_key, dim=1, eps=1e-8)
        neg_logit = F.cosine_similarity(x1=query, x2=neg_key, dim=1, eps=1e-8)

        loss = F.relu(input=self.margin+neg_logit-pos_logit)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                "Invalid reduction. Please choose from {{'mean', 'sum'}}."
            )

        return loss


def build_criterion(cfg, ignore_index: int) -> nn.Module:
    name = cfg.CRITERION.NAME.lower()

    criterion = None
    if name == 'crossentropy':
        criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=cfg.CRITERION.CROSSENTROPY.LABEL_SMOOTHING,
    )
    elif name == 'infonce':
        criterion = InfoNCE(
            temperature=cfg.CRITERION.INFONCE.TEMPERATURE,
            reduction=cfg.CRITERION.INFONCE.REDUCTION,
        )
    elif name == 'simcse':
        criterion = ContrastiveLoss(
            margin=cfg.CRITERION.CL.MARGIN,
            reduction=cfg.CRITERION.CL.REDUCTION,
        )
    elif name == "maxsim":
        criterion = MaxSim(
            temperature=cfg.CRITERION.MAXSIM.TEMPERATURE,
            reduction=cfg.CRITERION.MAXSIM.REDUCTION,
        )
    elif name == 'contrastive':
        criterion = ContrastiveLoss(
            margin=cfg.CRITERION.CONTRASTIVE.MARGIN,
            reduction=cfg.CRITERION.CONTRASTIVE.REDUCTION,
        )
    else:
        raise ValueError(
            "Invalid criterion. "
            "Please choose from {{'CrossEntropy', 'InfoNCE', 'SimCSE', 'MaxSim', 'Contrastive'}}."
        )

    return criterion
