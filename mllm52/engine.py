"""Autocomplete engine over a :class:`CausalTopology`."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from .topology import CausalTopology

_FLOOR = 1e-5
_UNIGRAM_WEIGHT = 0.1
_BACKOFF_SIZE = 50


@dataclass
class CompleteResult:
    """Outcome of an autocomplete run."""

    prefix: list[str]
    continuation: list[str]
    confidences: list[float]
    full_sequence: list[str]


class AutocompleteEngine:
    """Left-to-right autocomplete using causal n-gram statistics.

    Args:
        topo: Trained causal topology.
        rng: Random source. Pass ``random.Random(seed)`` for reproducible
            output; defaults to a nondeterministic instance.
    """

    def __init__(self, topo: CausalTopology, rng: random.Random | None = None) -> None:
        self.topo = topo
        self.rng = rng if rng is not None else random.Random()
        self._backoff = [w for w, _ in topo.unigrams.most_common(_BACKOFF_SIZE)]

    def _causal_contexts(self, seq: Sequence[str], idx: int):
        for n in range(1, self.topo.max_n + 1):
            if idx >= n:
                ctx = tuple(seq[idx - n : idx])
                yield n, ctx

    def causal_distribution(self, seq: Sequence[str], idx: int) -> dict[str, float]:
        """Left-only softmax distribution for position *idx* given *seq*."""
        base = 0.0
        contrib: dict[str, float] = {}
        for n, ctx in self._causal_contexts(seq, idx):
            counts = self.topo.left_counts[n].get(ctx)
            total = self.topo.left_totals[n].get(ctx, 0)
            if not counts or total <= 0:
                continue
            base += math.log(_FLOOR) * n
            floor_adj = math.log(_FLOOR) * n
            for word, cnt in counts.items():
                contrib[word] = contrib.get(word, 0.0) + (
                    math.log(cnt / total + _FLOOR) * n - floor_adj
                )
        candidates = sorted(set(contrib) | set(self._backoff))
        energies = {
            w: math.log(self.topo.unigrams.get(w, 1) + 1) * _UNIGRAM_WEIGHT
            + base
            + contrib.get(w, 0.0)
            for w in candidates
        }
        max_e = max(energies.values())
        exps = {w: math.exp(e - max_e) for w, e in energies.items()}
        total = sum(exps.values())
        return {w: e / total for w, e in exps.items()}

    def complete(
        self,
        prefix: Sequence[str],
        max_tokens: int = 16,
        temperature: float = 0.35,
        threshold: float = 0.0,
    ) -> CompleteResult:
        """Generate a causal continuation for *prefix*."""
        seq = [w.lower() for w in prefix]
        confidences: list[float] = []
        continuation: list[str] = []
        for _ in range(max_tokens):
            idx = len(seq)
            probs = self.causal_distribution(seq, idx)
            if not probs:
                break
            if temperature <= 0:
                chosen = max(probs, key=probs.get)  # type: ignore[arg-type]
                conf = probs[chosen]
            else:
                words = list(probs)
                weights = [p ** (1.0 / max(temperature, 1e-6)) for p in probs.values()]
                chosen = self.rng.choices(words, weights=weights, k=1)[0]
                conf = probs[chosen]
            if conf < threshold:
                break
            seq.append(chosen)
            continuation.append(chosen)
            confidences.append(conf)
            if chosen in {".", "!", "?"} and len(continuation) >= 4:
                break
        return CompleteResult(
            prefix=list(prefix),
            continuation=continuation,
            confidences=confidences,
            full_sequence=seq,
        )
