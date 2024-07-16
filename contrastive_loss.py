from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, tau: float) -> None:
        super().__init__()
        self.tau = tau
        return

    def forward(self, e, e_pos, e_neg) -> Tensor:
        cos_pos = F.cosine_similarity(x1=e, x2=e_pos, dim=1, eps=1e-8)
        cos_neg = F.cosine_similarity(x1=e, x2=e_neg, dim=1, eps=1e-8)

        numerator = torch.exp(cos_pos/self.tau)
        denominator = torch.exp(cos_pos/self.tau)+torch.exp(cos_neg/self.tau)

        loss = -torch.log(numerator/denominator).mean()

        return loss
