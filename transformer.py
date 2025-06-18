"""
Reference: https://pytorch.org/tutorials/beginner/translation_transformer.html
           https://github.com/hkproj/pytorch-transformer/blob/main
"""


from torch import Tensor
from typing import Optional

import logger
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int) -> None:
        """
        Args:
            vocab_size: vocabulary size
            dim: embedding dimension
        Returns:
            none
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input tensor [batch, seq_len]
        Returns:
            x: embeddings tensor [batch, seq_len, emb_dim]
        """
        # [B, L] -> [B, L, D]
        x = self.embeddings(x) * math.sqrt(self.dim)
        return x


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            seq_len: int,
            dim: int,
            dropout: float,
    ) -> None:
        """
        Args:
            seq_len: sequence length
            dim: embedding dimension
            dropout: dropout probability
        Returns:
            none
        """
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.dropout = nn.Dropout(p=dropout)

        # [L, D]
        pe = torch.zeros(size=(self.seq_len, self.dim))
        # [L, 1]
        pos = torch.arange(start=0, end=seq_len).reshape(seq_len, 1)
        # [D / 2]
        div_term = torch.exp(
            input=torch.arange(
                start=0,
                end=dim,
                step=2,
            ).float() * (-math.log(10000.0) / dim)
        )
        # sin(position * (10000 ** (2i / d_model))
        pe[:, 0::2] = torch.sin(input=pos * div_term)
        # cos(position * (10000 ** (2i / d_model))
        pe[:, 1::2] = torch.cos(input=pos * div_term)

        # [1, L, D]
        pe = pe.unsqueeze(dim=0)

        self.register_buffer(name='pe', tensor=pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input tensor [batch_size, seq_len, emb_dim]
        Returns:
            x: output tensor [batch_size, seq_len, emb_dim]
        """
        # [B, L, D]
        x += self.pe[:, :x.size(dim=1), :].requires_grad_(requires_grad=False)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        """
        Args:
            features: normalized shape
            eps: epsilon
        """
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(size=(features,)))
        self.bias = nn.Parameter(torch.zeros(size=(features,)))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input tensor [batch_size, seq_len, emb_dim]
        Returns:
            x: output tensor [batch_size, seq_len, emb_dim]
        """
        # [batch, seq_len, 1]
        mu = x.mean(dim=-1, keepdim=True)
        # [batch, seq_len, 1]
        sig = x.std(dim=-1, keepdim=True)
        x = self.alpha * (x - mu) / (sig + self.eps) + self.bias
        return x


class FFN(nn.Module):
    def __init__(self, dim: int, d_ff: int, dropout: float) -> None:
        """FeedForward Network
        Args:
            dim: embedding dimension
            d_ff: feedforward dimension
            dropout: dropout probability
        """
        super().__init__()
        self.up_proj = nn.Linear(
            in_features=dim,
            out_features=d_ff,
            bias=True,
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.down_proj = nn.Linear(
            in_features=d_ff,
            out_features=dim,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        # [B, L, D] -> [B, L, D_FF] -> [B, L, D]
        return self.down_proj(self.dropout(self.act(self.up_proj(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        """
        Args:
            dim: embedding dimension
            n_heads: number of heads
            dropout: dropout probability
        """
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        assert self.dim % self.n_heads == 0, (
            logger.log_error(
                f"{self.dim} is not divisible by {self.n_heads}"
            )
        )

        self.head_dim = dim // self.n_heads
        self.wq = nn.Linear(
            in_features=dim,
            out_features=dim,
            bias=False,
        )
        self.wk = nn.Linear(
            in_features=dim,
            out_features=dim,
            bias=False,
        )
        self.wv = nn.Linear(
            in_features=dim,
            out_features=dim,
            bias=False,
        )
        self.wo = nn.Linear(
            in_features=dim,
            out_features=dim,
            bias=False,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            mask: Optional[Tensor],
    ) -> Tensor:
        # [B, L_Q, D] -> [B, L_Q, H * D_H]
        q = self.wq(q)
        # [B, L_K, D] -> [B, L_K, H * D_H]
        k = self.wk(k)
        # [B, L_V, D] -> [B, L_V, H * D_H]
        v = self.wv(v)

        # [B, L_Q, H * D_H] -> [B, L_Q, H, D_H]
        q = q.view(q.size(dim=0), q.size(dim=1), self.n_heads, self.head_dim)
        # [B, L_K, H * D_H] -> [B, L_K, H, D_H]
        k = k.view(k.size(dim=0), k.size(dim=1), self.n_heads, self.head_dim)
        # [B, L_V, H * D_H] -> [B, L_V, H, D_H]
        v = v.view(v.size(dim=0), v.size(dim=1), self.n_heads, self.head_dim)

        # [B, L_Q, H, D_H] -> [B, H, L_Q, D_H]
        q = q.transpose(dim0=1, dim1=2)
        # [B, L_K, H, D_H] -> [B, H, L_K, D_H]
        k = k.transpose(dim0=1, dim1=2)
        # [B, L_V, H, D_H] -> [B, H, L_V, D_H]
        v = v.transpose(dim0=1, dim1=2)

        # [B, H, L_Q, D_H] @ [B, H, D_H, L_K] -> [B, H, L_Q, L_K]
        scores = q @ k.transpose(dim0=-2, dim1=-1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores.masked_fill(mask=~mask, value=float('-inf'))
        scores = F.softmax(scores, dim=-1)

        if self.dropout is not None:
            scores = self.dropout(scores)

        # [B, H, L_Q, L_K] @ [B, H, L_K, D_H] -> [B, H, L_Q, D_H]
        output = scores @ v

        # [B, H, L_Q, D_H] -> [B, L_Q, H, D_H] -> [B, L_Q, D]
        output = output.transpose(dim0=1, dim1=2).contiguous()\
            .view(output.size(dim=0), -1, self.n_heads * self.head_dim)

        # [B, L_Q, D] -> [B, L_Q, D]
        output = self.wo(output)

        return output


class EncoderBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            n_heads: int,
            d_ff: int,
            dropout: float,
    ) -> None:
        """
        Args:
            dim: embeddings dimension
            n_heads: number of heads
            d_ff: feedforward dimension
            dropout: dropout probability
        """
        super().__init__()
        self.attn_norm = LayerNorm(features=dim, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            dim=dim,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.attn_dropout = nn.Dropout(p=dropout)
        self.ffn_norm = LayerNorm(features=dim, eps=1e-6)
        self.ffn = FFN(
            dim=dim,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.ffn_dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        # self-attention
        x = x + self.attn_dropout(self._self_attn(self.attn_norm(x), mask=mask))
        # ffn
        x = x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))

        return x

    def _self_attn(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        return self.self_attn(q=x, k=x, v=x, mask=mask)


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, dim: int) -> None:
        """
        Args:
            layers: EncoderBlock
            dim: embeddings dimension
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features=dim, eps=1e-6)

    def forward(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            n_heads: int,
            d_ff: int,
            dropout: float,
    ) -> None:
        """
        Args:
            dim: embeddings dimension
            n_heads: number of heads
            d_ff: feedforward dimension
            dropout: dropout probability
        """
        super().__init__()
        self.self_attn_norm = LayerNorm(features=dim, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            dim=dim,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.self_attn_dropout = nn.Dropout(p=dropout)
        self.cross_attn_norm = LayerNorm(features=dim, eps=1e-6)
        self.cross_attn = MultiHeadAttention(
            dim=dim,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.cross_attn_dropout = nn.Dropout(p=dropout)
        self.ffn_norm = LayerNorm(features=dim, eps=1e-6)
        self.ffn = FFN(
            dim=dim,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.ffn_dropout = nn.Dropout(p=dropout)

    def forward(
            self,
            x: Tensor,
            tgt_attn_mask: Optional[Tensor],
            src_hidden_state: Tensor,
            src_attn_mask: Optional[Tensor],
    ) -> Tensor:
        # self-attention
        x = x + self.self_attn_dropout(
            self._self_attn(x=self.self_attn_norm(x), mask=tgt_attn_mask)
        )
        # cross-attention
        x = x + self.cross_attn_dropout(
            self._cross_attn(
                x=self.cross_attn_norm(x),
                hidden_state=src_hidden_state,
                mask=src_attn_mask,
            )
        )
        # ffn
        x = x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))

        return x

    def _self_attn(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        return self.self_attn(q=x, k=x, v=x, mask=mask)

    def _cross_attn(
            self,
            x: Tensor,
            hidden_state: Tensor,
            mask: Optional[Tensor],
    ) -> Tensor:
        return self.cross_attn(q=x, k=hidden_state, v=hidden_state, mask=mask)


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, dim: int) -> None:
        """
        Args:
            layers: DecoderBlocks
            dim: embeddings dimension
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features=dim, eps=1e-6)

    def forward(
            self,
            x: Tensor,
            tgt_attn_mask: Optional[Tensor],
            src_hidden_state: Tensor,
            src_attn_mask: Optional[Tensor],
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x=x,
                tgt_attn_mask=tgt_attn_mask,
                src_hidden_state=src_hidden_state,
                src_attn_mask=src_attn_mask,
            )
        x = self.norm(x)
        return x


class Projection(nn.Module):
    def __init__(self, dim: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features=dim, out_features=vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        # [B, L, vocab_size] -> [B, L, vocab_size]
        x = self.proj(x)
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            dim: int,
            src_vocab_size: int,
            tgt_vocab_size: int,
            src_seq_len: int,
            tgt_seq_len: int,
            n_heads: int,
            n_encoder_layers: int,
            n_decoder_layers: int,
            dim_feedforward: int,
            dropout: float,
    ) -> None:
        """
        Args:
            src_vocab_size: source vocabulary size
            tgt_vocab_size: target vocabulary size
            dim: embedding dimension
            src_seq_len: source sequence length
            tgt_seq_len: target sequence length
            n_encoder_layers: number of encoder layers
            n_decoder_layers: number of decoder layers
            n_heads: number of attention heads
            dropout: dropout probability
            dim_feedforward: feedforward dimension
        """
        super().__init__()
        self.src_tok_emb = Embedding(
            vocab_size=src_vocab_size,
            dim=dim,
        )
        self.src_pos_emb = PositionalEncoding(
            seq_len=src_seq_len,
            dim=dim,
            dropout=dropout,
        )
        self.tgt_tok_emb = Embedding(
            vocab_size=tgt_vocab_size,
            dim=dim
        )
        self.tgt_pos_emb = PositionalEncoding(
            seq_len=tgt_seq_len,
            dim=dim,
            dropout=dropout,
        )

        enc_blks = []
        for _ in range(n_encoder_layers):
            enc_blk = EncoderBlock(
                dim=dim,
                n_heads=n_heads,
                d_ff=dim_feedforward,
                dropout=dropout,
            )
            enc_blks.append(enc_blk)
        self.encoder = Encoder(
            layers=nn.ModuleList(modules=enc_blks),
            dim=dim,
        )

        dec_blks = []
        for _ in range(n_decoder_layers):
            dec_blk = DecoderBlock(
                dim=dim,
                n_heads=n_heads,
                d_ff=dim_feedforward,
                dropout=dropout,
            )
            dec_blks.append(dec_blk)
        self.decoder = Decoder(
            layers=nn.ModuleList(modules=dec_blks),
            dim=dim,
        )

        self.proj = Projection(dim=dim, vocab_size=tgt_vocab_size)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Reset parameters of the transformer."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(tensor=p)

    def encode(self, token_ids: Tensor, mask: Optional[Tensor]) -> Tensor:
        """
        Args:
            x: source tensor
            mask: source mask
        Returns:
            x: encoder output
        """
        x = self.src_tok_emb(x=token_ids)
        x = self.src_pos_emb(x=x)
        # [B, L] -> [B, 1, 1, L]
        mask = mask.unsqueeze(dim=1).unsqueeze(dim=1)
        x = self.encoder(x=x, mask=mask)

        return x

    def decode(
            self,
            tgt_token_ids: Tensor,
            tgt_attn_mask: Optional[Tensor],
            src_hidden_state: Tensor,
            src_attn_mask: Optional[Tensor],
    ) -> Tensor:
        """
        Args:
            x: target tensor
            mem: encoder output
            tgt_mask: target mask
            mem_mask: source mask
        Returns:
            x: decoder output
        """
        x = self.tgt_tok_emb(x=tgt_token_ids)
        x = self.tgt_pos_emb(x=x)
        # [B, L-1, L-1] -> # [B, 1, L-1, L-1]
        tgt_attn_mask = tgt_attn_mask.unsqueeze(dim=1)
        # [B, L] -> [B, 1, 1, L]
        src_attn_mask = src_attn_mask.unsqueeze(dim=1).unsqueeze(dim=1)
        x = self.decoder(
            x=x,
            tgt_attn_mask=tgt_attn_mask,
            src_hidden_state=src_hidden_state,
            src_attn_mask=src_attn_mask,
        )

        return x

    def project(self, x: Tensor) -> Tensor:
        x = self.proj(x)

        return x

    def forward(
            self,
            src_token_ids: Tensor,
            src_attn_mask: Optional[Tensor],
            tgt_token_ids: Tensor,
            tgt_attn_mask: Optional[Tensor],
    ) -> Tensor:
        src_hidden_state = self.encode(
            token_ids=src_token_ids,
            mask=src_attn_mask,
        )
        tgt_hidden_state = self.decode(
            tgt_token_ids=tgt_token_ids,
            tgt_attn_mask=tgt_attn_mask,
            src_hidden_state=src_hidden_state,
            src_attn_mask=src_attn_mask,
        )
        x = self.project(x=tgt_hidden_state)

        return x
