"""Causal n-gram topology (left context only) for MLLM-5.2."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"\b[a-zA-Z0-9']+\b|[.!?]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_RE = re.compile(r"\s+")


def tokenize(text: str) -> list[str]:
    """Split *text* into lowercase word and punctuation tokens.

    Words may contain letters, digits, and apostrophes; the punctuation
    marks ``.``, ``!``, and ``?`` become standalone tokens.
    """
    return TOKEN_RE.findall(text.lower())


class CausalTopology:
    """Causal n-gram statistics (left context only, n=1..max_n)."""

    def __init__(self, max_n: int = 3) -> None:
        if max_n < 1:
            raise ValueError(f"max_n must be >= 1, got {max_n}")
        self.max_n = max_n
        self.left_counts: dict[int, dict[tuple[str, ...], Counter[str]]] = {
            n: defaultdict(Counter) for n in range(1, max_n + 1)
        }
        self.left_totals: dict[int, dict[tuple[str, ...], int]] = {
            n: {} for n in range(1, max_n + 1)
        }
        self.unigrams: Counter[str] = Counter()
        self.vocab: set[str] = set()
        self.sentences: int = 0
        self.tokens: int = 0

    @classmethod
    def from_text(cls, text: str, max_n: int = 3) -> "CausalTopology":
        """Build a topology from a raw corpus string."""
        t = cls(max_n=max_n)
        t.ingest(text)
        return t

    def ingest(self, text: str) -> None:
        """Add *text* to the topology. May be called repeatedly.

        Raises:
            ValueError: If the text contains no usable tokens.
        """
        if not text or not text.strip():
            raise ValueError("corpus text is empty")
        flat = WHITESPACE_RE.sub(" ", text).strip()
        ingested = 0
        for sent in SENTENCE_SPLIT_RE.split(flat):
            words = tokenize(sent)
            if not words:
                continue
            self.sentences += 1
            ingested += len(words)
            self.vocab.update(words)
            self.unigrams.update(words)
            for n in range(1, self.max_n + 1):
                for i, tgt in enumerate(words):
                    if i >= n:
                        ctx = tuple(words[i - n : i])
                        self.left_counts[n][ctx][tgt] += 1
                        self.left_totals[n][ctx] = self.left_totals[n].get(ctx, 0) + 1
        self.tokens += ingested
        if ingested == 0:
            raise ValueError("corpus text contains no usable tokens")
