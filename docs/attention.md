# How attention works in this model

This is my own explanation of the attention mechanism, written while building the
transformer in [`rustgpt/model.py`](../rustgpt/model.py). The goal is to write down what
each piece actually does and why it is there, in terms I would use to explain it to a
colleague.

I deliberately kept two attention paths in the code. The default one calls PyTorch's
fused `scaled_dot_product_attention`, which is what you want for speed. The second one
(`use_sdpa=False` in `CausalSelfAttention.forward`) is the explicit version:
scores, mask, softmax, weighted sum, spelled out step by step. Everything below maps
directly onto that explicit path, so the math and the code line up one to one.


## The problem attention solves

A language model predicts the next token from the tokens before it. The hard part is
that the meaning of a token depends on other tokens that can be anywhere earlier in the
sequence. In Rust code, `self` only means something relative to the `impl` block it sits
in, a `?` operator only makes sense if the function returns a `Result`, and a closing
brace pairs with an opening one that might be hundreds of tokens back.

So each position needs a way to look at the other positions and pull in the information
that is relevant to it, right now. That is what attention is: a learned, content-based
lookup where every position decides which earlier positions matter and mixes their
information accordingly.


## Q, K, V: the three roles every token plays

For each token we project its embedding into three different vectors. In the code this
is one fused matrix that produces all three at once and then splits them:

```python
q, k, v = self.qkv(x).split(C, dim=2)
```

The three vectors play three distinct roles:

- **Query (Q)**: what this token is looking for. "I am a `?` operator, I want to find
  the function's return type."
- **Key (K)**: what this token offers as a match target. "I am a `Result<T, E>` return
  type, here is my advertisement."
- **Value (V)**: the actual information this token hands over once it has been matched.

The separation matters. A token can advertise itself one way (its key) while carrying
different content to pass along (its value), and it searches for matches using yet
another representation (its query). Three projections, three jobs, all learned.


## The core computation

The whole mechanism is one formula:

```
attention(Q, K, V) = softmax(Q Kᵀ / √d) V
```

Read left to right, this is what happens:

### 1. Scores: how well does each query match each key

`Q Kᵀ` is a dot product between every query and every key. The dot product is large when
two vectors point in a similar direction, so the score `Q[i] · K[j]` measures how
relevant token `j` is to token `i`. The result is a `(T, T)` matrix of scores, one row
per query, one column per key.

In the explicit code:

```python
att = (q @ k.transpose(-2, -1)) * (hd ** -0.5)   # (B, nh, T, T)
```

### 2. The √d scaling, and why it is not optional

That `* (hd ** -0.5)` is the `/ √d` in the formula, where `d` is the head dimension.
Here is the reason it exists. A dot product of two vectors of dimension `d` with
unit-ish components has variance that grows with `d`. So for a large head dimension the
raw scores get big, both positive and negative. Feed big numbers into a softmax and it
saturates: almost all the weight collapses onto a single position and the gradient
through the other positions goes to nearly zero. Training stalls. Dividing by `√d`
keeps the score variance roughly constant regardless of head size, which keeps the
softmax in a usable range. It is a normalization trick, not a cosmetic one.

### 3. The causal mask

This is a decoder, it generates left to right, so position `i` is only allowed to look at
positions `0..=i`. If it could see the future, training would be cheating: the model
would learn to read the answer instead of predicting it, and at generation time, where
the future genuinely does not exist, it would fall apart.

We enforce this by setting every score above the diagonal to negative infinity before
the softmax:

```python
mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
att = att.masked_fill(mask == 0, float("-inf"))
```

After the softmax those positions become exactly zero, so no information flows backward
in time. `softmax(-inf)` is `0`, which is exactly the behavior we want.

### 4. Softmax: scores become weights

Softmax turns each row of scores into a probability distribution: all weights positive,
each row sums to one.

```python
att = F.softmax(att, dim=-1)
```

Now row `i` says, in fractions that add up to 1, how much attention token `i` pays to
each earlier token. This is the part that makes attention content-based and dynamic:
the weights are computed fresh from the actual tokens, they are not fixed parameters.

### 5. The weighted sum of values

Finally we use those weights to mix the value vectors:

```python
out = att @ v
```

Each output position is a weighted average of the value vectors of the positions it
attended to. A token that needed its function's return type now has that information
blended into its own representation. That blended vector is what flows up to the next
layer.


## Multi-head attention

We do not run attention once over the full `n_embd` width. We split the width into
`n_head` smaller heads and run attention independently in each:

```python
q = q.view(B, T, nh, hd).transpose(1, 2)   # (B, nh, T, hd)
```

The reason is that one attention pattern per layer is too few. A single head is forced
to average all the different kinds of relationships into one set of weights. With
multiple heads, each one can specialize: one head can track bracket and brace matching,
another can link a variable to where it was declared, another can follow the
return-type relationship. They run in parallel over disjoint slices of the embedding,
then we concatenate the results and project them back to the model width:

```python
out = out.transpose(1, 2).contiguous().view(B, T, C)
return self.resid_dropout(self.proj(out))
```

Same total compute as one big head, far more expressive.


## Position: RoPE instead of learned embeddings

There is a subtlety hiding in everything above. The attention computation is permutation
equivariant: if you shuffle the tokens, the scores shuffle the same way, but nothing in
`Q Kᵀ` actually knows the order. Yet order is everything in code. `a - b` is not
`b - a`, and `let x = f()` must come before `x` is used.

The classic fix is to add a learned position embedding to each token. This model uses
rotary position embeddings (RoPE) instead, which is the modern choice. Rather than
adding a position signal, RoPE rotates the query and key vectors by an angle that
depends on their position, before the dot product:

```python
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)
```

The neat property is that after rotating `q` at position `i` and `k` at position `j`,
their dot product depends only on the relative offset `i - j`, not on the absolute
positions. So attention naturally learns relationships like "the token three positions
back" in a way that transfers across the whole sequence. The rotation angles are
precomputed once as `cos`/`sin` tables in `build_rope_cache` and reused every layer.


## Why the residual stream matters

One last thing that is easy to miss. Attention does not replace the token
representation, it adds to it:

```python
x = x + self.attn(self.ln1(x), cos, sin)   # communication
x = x + self.ffwd(self.ln2(x))             # computation
```

The `x + ...` is the residual connection. Attention writes its result into a running
sum that every layer reads from and adds to. I think of attention as the communication
step (tokens exchange information) and the feed-forward as the computation step (each
token processes what it gathered, on its own). Stacking many of these blocks lets
information travel across the sequence and get refined, layer after layer, which is how
the model builds up from raw tokens to something that can predict the next one well.


## The fused path

In production the explicit five-step path is replaced by a single call:

```python
out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=...)
```

This computes the exact same thing but never materializes the full `(T, T)` score matrix
in memory. It fuses the scale, mask, softmax, and weighted sum into one kernel
(flash-attention style), which is both faster and far lighter on memory, especially at a
1024-token context. The explicit version stays in the codebase because it is the one you
can read and reason about, and the two were checked to produce the same output to within
floating point noise.
