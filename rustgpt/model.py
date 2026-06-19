"""A modern decoder-only transformer (Llama-style), from scratch in PyTorch.

Upgrades over a vanilla GPT-2:
  - RoPE (rotary position embeddings) instead of learned absolute positions.
  - SwiGLU gated feed-forward instead of a ReLU MLP.
  - RMSNorm (pre-norm) instead of LayerNorm.
  - Weight tying between the token embedding and the output head.
  - Fused QKV, bias-free linears, scaled residual-projection init.

Attention runs through PyTorch's fused `scaled_dot_product_attention` (flash /
memory-efficient kernels) by default; a hand-rolled path (`use_sdpa=False`) is
kept as the readable reference for the attention write-up. Both apply RoPE
identically, so they produce the same result.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean-centering, no bias)."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_rope_cache(seq_len, head_dim, base=10000.0):
    """Precompute (cos, sin) of shape (seq_len, head_dim) for rotary embeddings."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (hd/2,)
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)             # (T, hd/2)
    emb = torch.cat((freqs, freqs), dim=-1)      # (T, hd)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    """Rotate q/k by their position. x: (B, nh, T, hd); cos/sin: (T, hd)."""
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return x * cos + rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE.

    `use_sdpa=True` (default) uses the fused kernel (no (B,nh,T,T) matrix in
    memory). `use_sdpa=False` is the explicit scores -> mask -> softmax -> sum
    reference path.
    """

    def __init__(self, n_embd, n_head, dropout=0.0, use_sdpa=True):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.dropout = dropout
        self.use_sdpa = use_sdpa
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)   # fused q, k, v
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        nh, hd = self.n_head, self.head_dim
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, nh, hd).transpose(1, 2)   # (B, nh, T, hd)
        k = k.view(B, T, nh, hd).transpose(1, 2)
        v = v.view(B, T, nh, hd).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.use_sdpa:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (hd ** -0.5)   # (B, nh, T, T)
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            out = att @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))


class SwiGLU(nn.Module):
    """Gated feed-forward: down(silu(gate(x)) * up(x)).

    Hidden size ~ (2/3)*4*n_embd so the 3-matrix SwiGLU matches the param budget
    of a 2-matrix 4*n_embd ReLU MLP.
    """

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        hidden = int(2 / 3 * 4 * n_embd)
        hidden = 64 * ((hidden + 63) // 64)   # round up to a multiple of 64
        self.w_gate = nn.Linear(n_embd, hidden, bias=False)
        self.w_up = nn.Linear(n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)), then x + ffwd(norm(x))."""

    def __init__(self, n_embd, n_head, dropout=0.0, use_sdpa=True):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, use_sdpa)
        self.ln2 = RMSNorm(n_embd)
        self.ffwd = SwiGLU(n_embd, dropout)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)   # communication
        x = x + self.ffwd(self.ln2(x))             # computation
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, context_size=1024,
                 dropout=0.0, use_sdpa=True):
        super().__init__()
        self.context_size = context_size
        head_dim = n_embd // n_head

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, dropout, use_sdpa) for _ in range(n_layer)]
        )
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight  # weight tying

        cos, sin = build_rope_cache(context_size, head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scale residual-projection inits by 1/sqrt(2*n_layer) (GPT-2 trick)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * n_layer) ** 0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.drop(self.token_embedding_table(idx))   # (B, T, C)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.ln_f(x)
        return self.lm_head(x)                            # (B, T, vocab_size)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively extend `idx` (B,T) by `max_new_tokens` sampled tokens."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_size:]       # never exceed the context window
            logits = self(idx_cond)[:, -1, :]            # (B, vocab) last step only
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
