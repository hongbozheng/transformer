"""
Reference: https://pytorch.org/tutorials/beginner/translation_transformer.html
           https://github.com/hkproj/pytorch-transformer/blob/main
"""


from typing import Tuple
from torch import Tensor
from typing import Optional
import logger
import torch
import torch.nn as nn
import math


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int) -> None:
        """
        Args:
            vocab_size: vocabulary size
            emb_dim: embedding dimension
        Returns:
            none
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
        )
        return

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input tensor [batch, seq_len]
        Returns:
            x: embeddings tensor [batch, seq_len, emb_dim]
        """
        # [batch, seq_len] -> [batch, seq_len, emb_dim]
        x = self.embeddings(x) * math.sqrt(self.emb_dim)
        return x


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            seq_len: int,
            emb_dim: int,
            dropout: float,
    ) -> None:
        """
        Args:
            seq_len: sequence length
            emb_dim: embedding dimension
            dropout: dropout probability
        Returns:
            none
        """
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(p=dropout)

        # [seq_len, emb_dim]
        pe = torch.zeros(size=(self.seq_len, self.emb_dim))
        # [seq_len, 1]
        pos = torch.arange(start=0, end=seq_len).reshape(seq_len, 1)
        # [emb_dim / 2]
        div_term = torch.exp(
            input=torch.arange(
                start=0,
                end=emb_dim,
                step=2,
            ).float() * (-math.log(10000.0) / emb_dim)
        )
        # sin(position * (10000 ** (2i / d_model))
        pe[:, 0::2] = torch.sin(input=pos * div_term)
        # cos(position * (10000 ** (2i / d_model))
        pe[:, 1::2] = torch.cos(input=pos * div_term)

        # [1, seq_len, emb_dim]
        pe = pe.unsqueeze(dim=0)

        self.register_buffer(name='pe', tensor=pe)
        return

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input tensor [batch_size, seq_len, emb_dim]
        Returns:
            x: output tensor [batch_size, seq_len, emb_dim]
        """
        # [batch, seq_len, emb_dim]
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
        return

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
    def __init__(self, emb_dim: int, d_ff: int, dropout: float) -> None:
        """FeedForward Network
        Args:
            emb_dim: embedding dimension
            d_ff: feedforward dimension
            dropout: dropout probability
        """
        super().__init__()
        self.linear_1 = nn.Linear(
            in_features=emb_dim,
            out_features=d_ff,
            bias=True,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear_2 = nn.Linear(
            in_features=d_ff,
            out_features=emb_dim,
            bias=True,
        )
        return

    def forward(self, x: Tensor) -> Tensor:
        # [batch, seq_len, emb_dim] -> [batch, seq_len, d_ff]
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # [batch, seq_len, d_ff] -> [batch, seq_len, emb_dim]
        x = self.linear_2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, dropout: float) -> None:
        """
        Args:
            emb_dim: embedding dimension
            n_heads: number of heads
            dropout: dropout probability
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        assert self.emb_dim % self.n_heads == 0, (
            logger.log_error(
                f"{self.emb_dim} is not divisible by {self.n_heads}"
            )
        )

        self.head_dim = emb_dim // self.n_heads
        self.w_q = nn.Linear(
            in_features=emb_dim,
            out_features=emb_dim,
            bias=False,
        )
        self.w_k = nn.Linear(
            in_features=emb_dim,
            out_features=emb_dim,
            bias=False,
        )
        self.w_v = nn.Linear(
            in_features=emb_dim,
            out_features=emb_dim,
            bias=False,
        )

        self.w_o = nn.Linear(
            in_features=emb_dim,
            out_features=emb_dim,
            bias=False,
        )
        self.dropout = nn.Dropout(p=dropout)
        return

    @staticmethod
    def attention(
            q: Tensor,
            k: Tensor,
            v: Tensor,
            mask: Optional[Tensor],
            dropout: Optional[nn.Dropout],
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            q: query   [batch, n_heads, seq_len, head_dim]
            k: key     [batch, n_heads, seq_len, head_dim]
            v: value   [batch, n_heads, seq_len, head_dim]
            mask: mask [batch, seq_len]
            dropout: dropout probability
        Returns:
            [batch, n_heads, seq_len, head_dim],
            [batch, n_heads, seq_len, seq_len]
        """
        head_dim = q.size(dim=-1)

        # [batch, n_heads, seq_len, seq_len]
        attn_scores = (q @ k.transpose(dim0=-2, dim1=-1)) / math.sqrt(head_dim)
        if mask is not None:
            attn_scores.masked_fill_(mask=mask == 0, value=float('-inf'))
        attn_scores = attn_scores.softmax(dim=-1)
        print(attn_scores)
        print(attn_scores.size())
        if dropout is not None:
            attn_scores = dropout(attn_scores)
        # [batch, n_heads, seq_len, head_dim]
        return (attn_scores @ v), attn_scores

    def forward(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        # [batch, seq_len, emb_dim] -> [batch, seq_len, emb_dim]
        q = self.w_q(q)
        # [batch, seq_len, emb_dim] -> [batch, seq_len, emb_dim]
        k = self.w_k(k)
        # [batch, seq_len, emb_dim] -> [batch, seq_len, emb_dim]
        v = self.w_v(v)

        # [batch, seq_len, emb_dim] -> [batch, seq_len, n_heads, head_dim]
        #                           -> [batch, n_heads, seq_len, head_dim]
        q = q.view(
            q.size(dim=0),
            q.size(dim=1),
            self.n_heads,
            self.head_dim,
        ).transpose(dim0=1, dim1=2)
        k = k.view(
            k.size(dim=0),
            k.size(dim=1),
            self.n_heads,
            self.head_dim,
        ).transpose(dim0=1, dim1=2)
        v = v.view(
            v.size(dim=0),
            v.size(dim=1),
            self.n_heads,
            self.head_dim,
        ).transpose(dim0=1, dim1=2)

        x, self.attn_scores = MultiHeadAttention.attention(
            q=q,
            k=k,
            v=v,
            mask=mask,
            dropout=self.dropout,
        )

        # [batch, n_heads, seq_len, head_dim] ->
        # [batch, seq_len, n_heads, head_dim]
        x = x.transpose(dim0=1, dim1=2)
        # [batch, seq_len, n_heads, head_dim] -> [batch, seq_len, emb_dim]
        x = x.contiguous().view(x.size(dim=0), -1, self.n_heads * self.head_dim)
        # [batch, seq_len, emb_dim]
        x = self.w_o(x)
        return x


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        """
        Args:
            features: embeddings dimension
            dropout: dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(features=features, eps=1e-6)
        return

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        _x = self.norm(x)
        _x = sublayer(_x)
        _x = self.dropout(_x)
        _x += x
        return _x


class EncoderBlock(nn.Module):
    def __init__(
            self,
            self_attn: MultiHeadAttention,
            ffn: FFN,
            features: int,
            dropout: float,
    ) -> None:
        """
        Args:
            self_attn: MultiHeadAttention
            ffn: FFN
            features: embeddings dimension
            dropout: dropout probability
        """
        super().__init__()
        self.self_attn = self_attn
        self.ffn = ffn
        self.residual = nn.ModuleList(
            [
                ResidualConnection(features=features, dropout=dropout)
                for _ in range(2)
            ]
        )
        self.dropout = nn.Dropout(p=dropout)
        return

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.residual[0](
            x=x,
            sublayer=lambda x: self.self_attn(q=x, k=x, v=x, mask=mask)
        )
        x = self.residual[1](x=x, sublayer=self.ffn)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, features: int) -> None:
        """
        Args:
            layers: EncoderBlock
            features: embeddings dimension
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features=features, eps=1e-6)
        return

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
            self,
            self_attn: MultiHeadAttention,
            cross_attn: MultiHeadAttention,
            ffn: FFN,
            features: int,
            dropout: float,
    ) -> None:
        """
        Args:
            self_attn: MultiHeadAttention
            cross_attn: MultiHeadAttention
            ffn: FFN
            features: embeddings dimension
            dropout: dropout probability
        """
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ffn = ffn
        self.residual = nn.ModuleList(
            [
                ResidualConnection(features=features, dropout=dropout)
                for _ in range(3)
            ]
        )
        return

    def forward(
            self,
            x: Tensor,
            memory: Tensor,
            tgt_mask: Tensor,
            mem_mask: Tensor,
    ) -> Tensor:
        x = self.residual[0](
            x=x,
            sublayer=lambda x: self.self_attn(q=x, k=x, v=x, mask=tgt_mask)
        )
        x = self.residual[1](
            x=x,
            sublayer=lambda x: self.cross_attn(
                q=x,
                k=memory,
                v=memory,
                mask=mem_mask,
            )
        )
        x = self.residual[2](x=x, sublayer=self.ffn)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, features: int) -> None:
        """
        Args:
            layers: DecoderBlocks
            features: embeddings dimension
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features=features, eps=1e-6)
        return

    def forward(
            self,
            x: Tensor,
            memory: Tensor,
            tgt_mask: Tensor,
            mem_mask: Tensor,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x=x,
                memory=memory,
                tgt_mask=tgt_mask,
                mem_mask=mem_mask,
            )
        x = self.norm(x)
        return x


class Projection(nn.Module):
    def __init__(self, emb_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features=emb_dim, out_features=vocab_size)
        return

    def forward(self, x: Tensor) -> Tensor:
        # [batch, seq_len, emb_dim] -> [batch, seq_len, vocab_size]
        x = self.proj(x)
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            emb_dim: int,
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
            emb_dim: embedding dimension
            src_seq_len: source sequence length
            tgt_seq_len: target sequence length
            n_encoder_layers: number of encoder layers
            n_decoder_layers: number of decoder layers
            n_heads: number of attention heads
            dropout: dropout probability
            dim_feedforward: feedforward dimension
        """
        super().__init__()
        self.src_emb = Embedding(
            vocab_size=src_vocab_size,
            emb_dim=emb_dim
        )
        self.src_pe = PositionalEncoding(
            seq_len=src_seq_len,
            emb_dim=emb_dim,
            dropout=dropout,
        )
        self.tgt_emb = Embedding(
            vocab_size=tgt_vocab_size,
            emb_dim=emb_dim
        )
        self.tgt_pe = PositionalEncoding(
            seq_len=tgt_seq_len,
            emb_dim=emb_dim,
            dropout=dropout,
        )

        encoder_blocks = []
        for _ in range(n_encoder_layers):
            self_attn = MultiHeadAttention(
                emb_dim=emb_dim,
                n_heads=n_heads,
                dropout=dropout,
            )
            ffn = FFN(
                emb_dim=emb_dim,
                d_ff=dim_feedforward,
                dropout=dropout,
            )
            encoder_block = EncoderBlock(
                self_attn=self_attn,
                ffn=ffn,
                features=emb_dim,
                dropout=dropout,
            )
            encoder_blocks.append(encoder_block)
        self.encoder = Encoder(
            layers=nn.ModuleList(modules=encoder_blocks),
            features=emb_dim,
        )

        decoder_blocks = []
        for _ in range(n_decoder_layers):
            self_attn = MultiHeadAttention(
                emb_dim=emb_dim,
                n_heads=n_heads,
                dropout=dropout,
            )
            cross_attn = MultiHeadAttention(
                emb_dim=emb_dim,
                n_heads=n_heads,
                dropout=dropout,
            )
            ffn = FFN(
                emb_dim=emb_dim,
                d_ff=dim_feedforward,
                dropout=dropout,
            )
            decoder_block = DecoderBlock(
                self_attn=self_attn,
                cross_attn=cross_attn,
                ffn=ffn,
                features=emb_dim,
                dropout=dropout,
            )
            decoder_blocks.append(decoder_block)
        self.decoder = Decoder(
            layers=nn.ModuleList(modules=decoder_blocks),
            features=emb_dim,
        )

        self.proj = Projection(emb_dim=emb_dim, vocab_size=tgt_vocab_size)

        self._reset_parameters()

        return

    def _reset_parameters(self) -> None:
        """Reset parameters of the transformer."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(tensor=p)
        return

    def encode(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x: source tensor
            mask: source mask
        Returns:
            x: encoder output
        """
        x = self.src_emb(x=x)
        x = self.src_pe(x=x)
        x = self.encoder(x=x, mask=mask)
        return x

    def decode(
            self,
            x: Tensor,
            memory: Tensor,
            tgt_mask: Tensor,
            mem_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            x: target tensor
            memory: encoder output
            tgt_mask: target mask
            mem_mask: source mask
        Returns:
            x: decoder output
        """
        x = self.tgt_emb(x=x)
        x = self.tgt_pe(x=x)
        x = self.decoder(
            x=x,
            memory=memory,
            tgt_mask=tgt_mask,
            mem_mask=mem_mask,
        )
        return x

    def project(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x

    def forward(
            self,
            src: Tensor,
            tgt: Tensor,
            src_mask: Tensor,
            tgt_mask: Tensor,
    ) -> Tensor:
        memory = self.encode(x=src, mask=src_mask)
        x = self.decode(
            x=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            mem_mask=src_mask,
        )
        x = self.project(x=x)
        return x