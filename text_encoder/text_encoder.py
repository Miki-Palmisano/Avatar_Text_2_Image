"""
Small Transformer text encoder trained from scratch (no pretrained weights,
no CLIP/BERT/T5 — per the assignment's mandatory constraints).

Produces per-token hidden states (B, T, D) for cross-attention conditioning
in the U-Net, plus a pooled (B, D) vector if you'd rather condition via
FiLM/addition instead of cross-attention.

A learned "null" embedding sequence is used whenever a sample's text
condition is dropped (see cond_mask below). This is what lets you train
BOTH the unconditional baseline and the conditional model with the exact
same architecture/code path: for the unconditional run you simply always
drop the condition (uncond_prob=1.0); for the conditional run you drop it
with a small probability (e.g. 0.1) for classifier-free-guidance-style
robustness, or never drop it if you don't want CFG.
"""
import math
import torch
import torch.nn as nn


class SinusoidalOrLearnedPositional(nn.Module):
    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)

    def forward(self, x):
        T = x.shape[1]
        return x + self.pos_emb[:, :T, :]


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int = 24,
        dim: int = 96,
        n_layers: int = 3,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.pad_id = pad_id

        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.pos_emb = SinusoidalOrLearnedPositional(max_len, dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.ln_out = nn.LayerNorm(dim)

        # learned embedding used for "dropped"/unconditional samples,
        # broadcast across the sequence length.
        self.null_embedding = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, input_ids: torch.Tensor, cond_mask: torch.Tensor):
        """
        input_ids: (B, T) long
        cond_mask: (B,) bool/float — True/1 = use real text, False/0 = use null embedding
        returns:
            hidden_states: (B, T, D)  -- for cross-attention
            pooled:        (B, D)     -- mean-pooled over non-pad tokens, for FiLM/add conditioning
        """
        B, T = input_ids.shape
        pad_mask = input_ids.eq(self.pad_id)  # (B, T) True where padding

        x = self.token_emb(input_ids)          # (B, T, D)
        x = self.pos_emb(x)
        hidden = self.encoder(x, src_key_padding_mask=pad_mask)
        hidden = self.ln_out(hidden)

        cond_mask = cond_mask.view(B, 1, 1).to(hidden.dtype)
        null = self.null_embedding.expand(B, T, self.dim)
        hidden = cond_mask * hidden + (1 - cond_mask) * null

        valid = (~pad_mask).float().unsqueeze(-1)                   # (B,T,1)
        pooled = (hidden * valid).sum(1) / valid.sum(1).clamp(min=1.0)

        return hidden, pooled