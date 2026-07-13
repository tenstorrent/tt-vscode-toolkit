# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# ============================================================================
# The "build it yourself" half of Lab 1: a minimal, from-scratch byte-pair-
# encoding (BPE) tokenizer. No external tokenizer libraries (no `tiktoken`,
# no Hugging Face `tokenizers`) -- the entire point of this file is that you
# can read every line of the algorithm that turns text into token IDs.
#
# This is the tokenizer you'd reach for at ~80M-parameter / TinyStories scale
# (see lfs-00's "honest runtime matrix" and two-tier promise). The nano model
# actually trained on Blackhole in Lab 5 uses the even simpler CharTokenizer
# in reference_gpt.py -- one integer per character, no merges at all. Both
# are shown side by side in lfs-01-tokenizer.md so the two-tier story stays
# honest: this file is real, working BPE; it just isn't the tokenizer behind
# the nano run.
#
# Algorithm (Sennrich et al. 2016, "Neural Machine Translation of Rare Words
# with Subword Units"; same core loop as Karpathy's minbpe):
#   1. Start with a vocabulary of the 256 raw bytes -- every possible byte is
#      already representable, so BPE never needs an "unknown token".
#   2. Count every adjacent pair of tokens across the training corpus.
#   3. Merge the single most frequent pair into one new token, add it to the
#      vocab, and repeat for a fixed number of merges.
#   4. encode() replays the learned merges, in the order they were learned,
#      against new text. decode() just concatenates token bytes back together
#      and decodes as UTF-8.
# ============================================================================
"""Minimal from-scratch byte-pair-encoding (BPE) tokenizer.

Self-test (train on a tiny Shakespeare excerpt, then round-trip a string):

    python content/templates/llm-from-scratch/tokenizer_bpe.py

Expected: prints vocab size, a handful of learned merges, and "PASSED".
"""

from collections import Counter

# A tiny excerpt of the public-domain "tiny Shakespeare" corpus -- the same
# corpus ttml's from-scratch training run in Lab 5 char-tokenizes. Kept
# deliberately small: this file has no data dependency beyond this constant,
# trains in well under a second, and still has enough repeated character
# pairs ("Citizen", "speak", "know") to make merges worth learning.
SAMPLE_TEXT = """\
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
"""


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

    @staticmethod
    def _count_pairs(ids: list[int]) -> Counter:
        """Count how often each adjacent (id, id) pair occurs."""
        return Counter(zip(ids, ids[1:]))

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace every occurrence of `pair` in `ids` with `new_id`."""
        out = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2  # consumed both tokens of the pair
            else:
                out.append(ids[i])
                i += 1
        return out

    def encode(self, text: str) -> list[int]:
        """Tokenize new text by replaying learned merges in priority order."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            pair_counts = self._count_pairs(ids)
            # Of the pairs actually present, apply whichever one was learned
            # EARLIEST in training (lowest merge id) -- that's the BPE
            # merge-priority rule, and it's what keeps encode() deterministic.
            mergeable = [p for p in pair_counts if p in self.merges]
            if not mergeable:
                break  # nothing left that any learned merge applies to
            pair = min(mergeable, key=lambda p: self.merges[p])
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Concatenate token bytes back into a string."""
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")


def self_test() -> int:
    tok = BPETokenizer()
    tok.train(SAMPLE_TEXT, num_merges=50)

    learned = len(tok.merges)
    print(f"vocab size: {tok.vocab_size}  (256 raw bytes + {learned} learned merges)")
    print("first 5 learned merges:")
    for pair, new_id in list(tok.merges.items())[:5]:
        left, right = tok.vocab[pair[0]], tok.vocab[pair[1]]
        print(f"  {left!r} + {right!r} -> {tok.vocab[new_id]!r}  (id {new_id})")

    probe = "First Citizen: Speak, speak."
    ids = tok.encode(probe)
    round_trip = tok.decode(ids)
    print(f"\nprobe text: {probe!r}")
    print(f"encoded ({len(ids)} tokens): {ids}")
    print(f"decoded:    {round_trip!r}")
    assert round_trip == probe, "round-trip failed on training-adjacent text"

    # Prove it also round-trips text with bytes/pairs never seen in training
    # -- the whole reason byte-level BPE never hits an "unknown token".
    unseen = "Zzyzx 42 -- an em dash and digits never seen in training."
    assert tok.decode(tok.encode(unseen)) == unseen, "round-trip failed on unseen text"

    print("\nPASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test())
