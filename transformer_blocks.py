"""Building blocks for a pre-normalized decoder-only Transformer."""

import torch
from torch import nn


class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_dim, block_size, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.scale = head_size ** -0.5
        self.register_buffer("tril", torch.ones(block_size, block_size, dtype=torch.bool).tril())

    def forward(self, x):
        length = x.size(1)
        scores = (self.query(x) @ self.key(x).transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~self.tril[:length, :length], float("-inf"))
        return scores.softmax(dim=-1) @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, block_size, num_heads):
        super().__init__()
        if num_heads < 1 or embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by a positive num_heads")
        head_size = embedding_dim // num_heads
        self.heads = nn.ModuleList(
            SelfAttentionHead(embedding_dim, block_size, head_size) for _ in range(num_heads)
        )
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        return self.proj(torch.cat([head(x) for head in self.heads], dim=-1))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embedding_dim, block_size, n_heads):
        super().__init__()
        self.sa = MultiHeadAttention(embedding_dim, block_size, n_heads)
        self.ffwd = FeedForward(embedding_dim)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))
