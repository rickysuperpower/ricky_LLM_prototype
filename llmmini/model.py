import torch
import torch.nn as nn
from .config import GPTConfig
from .blocks import DummyTransformerBlock

class DummyGPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        self.final_norm = nn.LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, in_idx):
        # in_idx: (B, T)
        _, seq_len = in_idx.shape
        tok = self.tok_emb(in_idx)  # (B, T, D)
        pos = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # (T, D)
        x = tok + pos
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)  # (B, T, V)
        return logits
