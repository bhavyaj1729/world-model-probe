import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a                      # residual around attention
        x = x + self.mlp(self.ln2(x))  # residual around MLP
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab=29, d=64, heads=2, ff=128, layers=3, max_len=21):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, heads, ff) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x, return_hidden=False):
        n = x.shape[1]
        # True = blocked, so each token can only look at earlier ones
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.tok(x) + self.pos(torch.arange(n, device=x.device))
        hidden = [h]                   # saving every layer's output for probing later
        for b in self.blocks:
            h = b(h, mask)
            hidden.append(h)
        logits = self.head(self.ln_f(h))
        return (logits, hidden) if return_hidden else logits
