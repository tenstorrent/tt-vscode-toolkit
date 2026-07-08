---
id: lfs-01-tokenizer
title: "Tokenizer & Data from Scratch"
description: >-
  Build a BPE tokenizer and data pipeline from scratch, then see how a token
  sequence becomes tiled ttnn tensors on Tenstorrent hardware.
category: llm-from-scratch
tags:
  - tokenizer
  - bpe
  - data
  - tinystories
  - llm
supportedHardware: [n150, n300, t3k, p100, p150, p300c, galaxy, simulator]
status: draft
estimatedMinutes: 30
---

# Tokenizer & Data from Scratch

Every model in this arc starts the same way: text goes in, integers come
out. This lab builds that conversion **from scratch**, with no tokenizer
library doing the work for you, and follows the integers all the way to the
tiled `ttnn.Tensor` shape the rest of the arc builds on.

---

## Coming from CUDA: moving data on and off the device

Before any tokenizer code, one CUDA habit needs adjusting. On CUDA, a tensor
is (usually) either explicitly on the host or explicitly on the device, and
you move it with `cudaMemcpy`:

```c
cudaMemcpy(d_tokens, h_tokens, size, cudaMemcpyHostToDevice);   // H2D
cudaMemcpy(h_logits, d_logits, size, cudaMemcpyDeviceToHost);   // D2H
```

TT-NN<sup>™</sup>'s equivalent is a pair of Python calls, not a driver-level copy:

```python
tt_tokens = ttnn.from_torch(torch_tokens, device=device)   # H2D
torch_logits = ttnn.to_torch(tt_logits)                    # D2H
```

**There is no unified memory here, same as on discrete CUDA GPUs without
`cudaMallocManaged`.** A tensor is either a plain `torch.Tensor` living in
host RAM, or a `ttnn.Tensor` living in a Tensix chip's DRAM — never both at
once, and the move between them is always an explicit call you can see in
the code. Everything in this lab happens on the host side of that line: a
tokenizer is pure Python, and the first `ttnn.from_torch` doesn't happen
until the very end, once there's a batch of token IDs worth moving.

---

## Build a BPE tokenizer, from scratch

Real tokenizers almost never map one character to one token — that wastes
vocabulary slots on redundant sequences ("th", "ing", "the ") that show up
constantly in real text. **Byte-pair encoding (BPE)** fixes this by
*learning* which byte sequences are common enough to deserve their own
token, starting from nothing but the 256 raw bytes and merging upward.

The algorithm, in four steps:

1. Start with a vocabulary of the 256 raw bytes. Every possible byte is
   already representable, so BPE never hits an "unknown token" — text
   outside the training data just falls back to individual bytes.
2. Count every adjacent pair of tokens across the training corpus.
3. Merge the single most frequent pair into one new token, add it to the
   vocabulary, and repeat for a fixed number of merges.
4. To tokenize new text, replay the learned merges — in the order they were
   learned — until no more apply.

Here's the whole thing, quoted verbatim from
`content/templates/llm-from-scratch/tokenizer_bpe.py` so the prose and the
code never drift:

```python
class BPETokenizer:
    """Byte-level BPE: start from raw bytes, greedily merge frequent pairs.

    Every token bottoms out at raw bytes, so any UTF-8 string -- including
    text the tokenizer never saw during training -- round-trips exactly
    through decode(encode(text)). Unseen byte pairs just stay as individual
    byte tokens instead of merging; there is no "unknown token" case to
    handle, unlike a fixed word-level vocabulary.
    """

    def __init__(self):
        # vocab: token id -> the bytes it expands to. IDs 0-255 are always
        # the 256 raw bytes; IDs 256+ are merges learned by train().
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # merges: (id, id) pair -> the new id it merges into, keyed in the
        # ORDER they were learned. That order is the merge *priority*:
        # encode() must resolve ties the same way training discovered them,
        # or the same text could tokenize two different ways.
        self.merges: dict[tuple[int, int], int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, num_merges: int) -> None:
        """Learn up to `num_merges` merge rules from `text`."""
        ids = list(text.encode("utf-8"))  # start as one token per raw byte

        for i in range(num_merges):
            pair_counts = self._count_pairs(ids)
            if not pair_counts:
                break  # fewer than 2 tokens left -- nothing left to merge
            best_pair = max(pair_counts, key=pair_counts.get)
            new_id = 256 + i
            left, right = self.vocab[best_pair[0]], self.vocab[best_pair[1]]
            self.vocab[new_id] = left + right
            self.merges[best_pair] = new_id
            ids = self._merge(ids, best_pair, new_id)
```

`encode()` and `decode()` are the mirror image of `train()` — `encode()`
walks the same merge table forward on new text; `decode()` just concatenates
token bytes and decodes as UTF-8. The full file (`_merge`, `encode`,
`decode`, and a self-test) is about 130 lines, all pure Python, no
dependencies beyond `collections.Counter`.

Run it and you'll see merges being learned live, on a tiny Shakespeare
excerpt:

```
$ python content/templates/llm-from-scratch/tokenizer_bpe.py
vocab size: 306  (256 raw bytes + 50 learned merges)
first 5 learned merges:
  b'e' + b' ' -> b'e '  (id 256)
  b':' + b'\n' -> b':\n'  (id 257)
  b't' + b' ' -> b't '  (id 258)
  b'.' + b'\n' -> b'.\n'  (id 259)
  b'l' + b'l' -> b'll'  (id 260)

probe text: 'First Citizen: Speak, speak.'
encoded (9 tokens): [273, 58, 32, 83, 283, 265, 115, 283, 46]
decoded:    'First Citizen: Speak, speak.'

PASSED
```

Notice the first merges are boring on purpose — `"e "`, `":\n"`, `"t "` —
because BPE always finds the *globally* most frequent pair first, and in
English prose that's punctuation-plus-whitespace, not word fragments. Train
on a bigger corpus for longer and the merges climb toward whole common words
("the", "and") and then multi-word fragments. The probe sentence — 29
characters — collapses to 9 tokens once trained, because "First Citizen"
and "speak" both showed up (and got merged) during training.

**This is the tokenizer for the ~80M-parameter / TinyStories-scale target**
this arc frames as its "hero" run (see lfs-00) — and it's the same family of
tokenizer used by the r/LocalLLaMA build that inspired this whole arc. If
you wanted to point it at a real dataset instead of the tiny inline sample,
you'd pull it with the `hf` CLI, not `huggingface-cli`:

```bash
hf download roneneldan/TinyStories --repo-type dataset --local-dir ~/data/tinystories
```

Then swap `SAMPLE_TEXT` for the downloaded corpus and raise `num_merges`
from 50 to a few thousand — the algorithm doesn't change, only the corpus
and the merge budget do.

---

## The honest two-tier tokenizer story

Here's the part that's easy to gloss over, and this lesson won't: **the
nano model this arc actually trains on Blackhole<sup>®</sup> in Lab 5 does not use the
BPE tokenizer above.** It uses something simpler — `CharTokenizer`, already
sitting in `reference_gpt.py`:

```python
class CharTokenizer:
    """The simplest possible tokenizer: one integer per character.

    Mirrors ttml.common.data.CharTokenizer. A real LLM uses BPE (Lab 1 builds
    one), but char-level is enough to make the data pipeline concrete.
    """

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)
```

No merges, no training loop — just every distinct character in the corpus
assigned an integer. That's genuinely all `ttml`'s from-scratch training run
needs at nano scale (`vocab≈96` on tiny Shakespeare), and it's simple enough
that the data pipeline stays the star of the show rather than the
tokenizer. **Both tokenizers are real and both are used honestly in this
arc** — BPE is what you'd reach for at TinyStories/~80M scale (and what
this section builds), `CharTokenizer` is what actually runs, live, on
Blackhole hardware in Lab 5. Neither is a placeholder for the other.

---

## Data pipeline: from a token stream to training examples

Once you have `encode()`, a whole corpus becomes one long list of integers.
Training needs fixed-length windows into that list, batched together:

```python
import torch

def get_batch(token_ids: list[int], batch_size: int, block_size: int):
    """Sample `batch_size` random windows of length `block_size` from a
    token stream, each paired with itself shifted one position right — the
    standard "predict the next token" training pair.
    """
    data = torch.tensor(token_ids, dtype=torch.long)
    # Random starting offsets, leaving room for a full window + 1 target.
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y  # both [batch_size, block_size]
```

`x` is the model's input; `y` is `x` shifted one token to the right — at
every position, the target is simply "the next token." This is the exact
`[batch, seq]` shape (`batch=2, seq=32` at nano scale) that
`reference_gpt.py`'s `smoke()` function feeds into the model, and the shape
every downstream lab builds on. Nothing about batching is TT-specific —
this step is identical whether the next stop is `ttnn` or plain PyTorch.

---

## Tokens → tiled tensors

This is where the CUDA callout from the top of the lesson stops being
abstract. A `[batch, seq]` integer tensor is still a `torch.Tensor` living
on the host at this point — nothing has crossed onto a Tensix chip yet.
Getting it there is two explicit steps:

```python
import ttnn

# Host -> device, same move as cudaMemcpyHostToDevice.
tt_tokens = ttnn.from_torch(
    x,                          # the [batch, seq] int tensor from get_batch()
    device=device,
    layout=ttnn.TILE_LAYOUT,    # reshuffled into 32x32 tiles, not row-major
)
```

Recall from lfs-00: Tensix hardware doesn't compute on scalars or arbitrary
rows, it computes on **32×32 tiles** — 1,024 elements addressed as a single
unit. `ttnn.TILE_LAYOUT` is what performs that reshuffle. At nano scale
(`block_size=256`), a `[2, 256]` token tensor already divides evenly into
32-wide tiles along the sequence axis; padding only becomes a concern once
`seq` or the embedding width isn't a multiple of 32, which lfs-02 runs into
directly.

The token tensor itself is just IDs — it doesn't become a *meaningful*
tensor to the model until each ID is looked up in an embedding table and
turned into a `n_embd`-wide vector. That lookup is one call:

```python
# Preview only -- lfs-02 builds and owns this.
x_embedded = ttnn.embedding(tt_tokens, embedding_weight, layout=ttnn.TILE_LAYOUT)
```

`ttnn.embedding` is a gather: for every integer in `tt_tokens`, it pulls the
matching row out of `embedding_weight` and returns a `[batch, seq, n_embd]`
tensor, already tiled. That's real code from the next lab, not this one —
lfs-02 builds the embedding table and the residual stream it feeds into.
The one thing worth fixing in your head now: **the token tensor produced by
this lab's tokenizer is exactly the input `ttnn.embedding` expects.** The
seam between "data from scratch" and "model from scratch" is a single,
explicit `ttnn.from_torch` call.

---

## Run it

Everything in this lab is pure Python — no device, no simulator, no `ttnn`
import required to see it work:

```bash
python content/templates/llm-from-scratch/tokenizer_bpe.py
```

Expected: a printed vocab size (`306`), the first five learned merges, a
round-trip encode/decode of a probe sentence, a second round-trip on text
containing bytes never seen during training, and `PASSED` on exit 0. That's
the whole tokenizer, verified, before a single line of model code exists.

---

## Graduate box

Everything you just ran — training merges, encoding, decoding, batching —
runs on a laptop CPU with no Tenstorrent hardware, no simulator, and no
device driver in sight. That's deliberate: data and tokenization are a
host-side problem on every stack, CUDA included. **The token tensor lands
on-device for the first time in the next lab**, the moment `ttnn.from_torch`
hands a batch to `ttnn.embedding` — and that's also where your first
TT-Lang kernel gets written.

---

## Next

[→ Continue to Lab 2: Embeddings & the Residual Stream](command:tenstorrent.showLesson?["lfs-02-embeddings"])
