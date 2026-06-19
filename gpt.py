import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """A single self-attention head (the naive, one-head-at-a-time version)."""

    def __init__(self, head_size, n_embd, context_size=1024, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # tril is state, not a parameter -> register as a buffer so .to()/state_dict handle it
        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T), scaled dot-product
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # causal masking
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)  # randomly drop attention links during training
        v = self.value(x)  # (B, T, head_size)
        return wei @ v     # (B, T, head_size)


class OptimizedMultiHeadAttention(nn.Module):
    """Multi-head attention with all heads fused into single q/k/v projections."""

    def __init__(self, num_heads, head_size, n_embd, context_size=1024, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        # one projection per q/k/v, covering ALL heads at once
        self.key = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.query = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.value = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.attn_dropout = nn.Dropout(dropout)   # on the attention weights
        self.resid_dropout = nn.Dropout(dropout)  # on the output added back to the residual
        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))

    def forward(self, x):
        B, T, C = x.shape
        nh, hs = self.num_heads, self.head_size
        # (B,T,C) -> (B,T,nh,hs) -> (B,nh,T,hs): move the heads into the batch dim
        k = self.key(x).view(B, T, nh, hs).transpose(1, 2)
        q = self.query(x).view(B, T, nh, hs).transpose(1, 2)
        v = self.value(x).view(B, T, nh, hs).transpose(1, 2)

        wei = q @ k.transpose(-2, -1) * (hs ** -0.5)          # (B,nh,T,T), a single batched matmul
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)
        out = wei @ v                                          # (B,nh,T,hs)
        out = out.transpose(1, 2).contiguous().view(B, T, nh * hs)  # recombine the heads
        return self.resid_dropout(self.proj(out))


class MultiHeadAttention(nn.Module):
    """Naive multi-head attention: a list of independent heads, concatenated."""

    def __init__(self, num_heads, head_size, n_embd, context_size=1024, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, context_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),  # on the output added back to the residual
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """A transformer block: pre-norm attention + feed-forward, both residual."""

    def __init__(self, n_embd, n_head, context_size=1024, dropout=0.0):
        super().__init__()
        self.sa = OptimizedMultiHeadAttention(n_head, n_embd // n_head, n_embd, context_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # communication (pre-norm + residual)
        x = x + self.ffwd(self.ln2(x))  # computation  (pre-norm + residual)
        return x


class Transformer(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, context_size=1024, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, context_size, dropout) for _ in range(n_layer)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, context_size=1024, dropout=0.0):
        super().__init__()
        self.context_size = context_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_size, n_embd)
        self.drop = nn.Dropout(dropout)  # on the (token + position) embedding sum
        self.blocks = Transformer(n_embd, n_head, n_layer, context_size, dropout)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                          # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)
        x = self.drop(tok_emb + pos_emb)  # (B,T,C)
        x = self.blocks(x)     # (B,T,C)
        x = self.ln_f(x)       # (B,T,C)
        return self.lm_head(x)  # (B,T,vocab_size)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively extend `idx` (B,T) by `max_new_tokens` sampled tokens.

        Each step: crop the context to the last `context_size` tokens, take the
        logits at the final position, optionally sharpen (temperature) and clip to
        the top-k choices, sample one token, and append it.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_size:]      # never feed more than the context window
            logits = self(idx_cond)[:, -1, :]           # (B, vocab), only the last step predicts next
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
