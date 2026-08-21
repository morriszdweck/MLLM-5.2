#!/usr/bin/env python3
"""
MLLM-5.2 — Document Autocomplete Micro Language Model
Single-file, zero-dependency, pure Python 3.10+

A tiny autocomplete LM that continues your document.
Core idea: build a causal n-gram topology from a corpus, then complete
prefixes left-to-right. Each position is scored only on left context
(distances 1..3), sampled under temperature, gated by confidence — like
ghost-text in an editor. Tab to accept.

Usage
-----
  python MLLM-5.2.py "the quick brown"                # one-shot autocomplete (default)
  python MLLM-5.2.py autocomplete "hello world" --steps 16 --seed 42
  python MLLM-5.2.py generate "what is an atom" --steps 30 --seed 42  # alias
  python MLLM-5.2.py                                   # interactive autocomplete REPL
  python MLLM-5.2.py chat --steps 24                   # alias for REPL

Principles (autocomplete-focused)
----------------------------------
  • causal n-gram topology (left context only, n=1..3)
  • left-to-right generation: prefix frozen, continuation sampled step-by-step
  • temperature + threshold gating (ghost only if confident)
  • diffusion heritage: annealed steps still available via --steps/effort
  • pure stdlib, deterministic with --seed, works offline

No install needed. Just Python 3.10+. Optionally install for `mllm52` CLI
via `pip install -e .` (package re-exports this file).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import re
import sys
import textwrap
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

__version__ = "5.2"
MASK = "<mask>"

# ───────────────────────────────────────────────────────────── tokenization
TOKEN_RE = re.compile(r"\b[a-zA-Z0-9']+\b|[.!?]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_RE = re.compile(r"\s+")

def tokenize(text: str) -> list[str]:
    """Split *text* into lowercase word and punctuation tokens."""
    return TOKEN_RE.findall(text.lower())

# ───────────────────────────────────────────────────────────── embedded corpus
BUILT_IN_CORPUS = """
# MLLM-5.2 Preview — define your corpus here
# This corpus is autocomplete-based (ghost-text, left-to-right continuation examples).
# Replace this placeholder with your text. Example (delete):
# The quick brown fox jumps over the lazy dog.
# Hello world, this is autocomplete training data.
"""
# ───────────────────────────────────────────────────────────── terminal colors

_CODES = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
    "red": "\033[31m",
}
_RESET = "\033[0m"
_MASK_STYLED = "\033[41m\033[37m[MASK]\033[0m"


class Term:
    """ANSI painter that auto-disables when not a tty or NO_COLOR is set."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    @classmethod
    def detect(cls, no_color: bool = False, stream=None) -> "Term":
        if stream is None:
            stream = sys.stdout
        is_tty = hasattr(stream, "isatty") and stream.isatty()
        enabled = not no_color and is_tty and os.environ.get("NO_COLOR") is None
        return cls(enabled=enabled)

    def paint(self, color: str, text: str) -> str:
        if not self.enabled:
            return text
        code = _CODES.get(color, "")
        return f"{code}{text}{_RESET}" if code else text

    @property
    def mask(self) -> str:
        return _MASK_STYLED if self.enabled else "[MASK]"


# ───────────────────────────────────────────────────────────── topology

class BidirectionalTopology:
    """Bidirectional n-gram statistics over a corpus.

    For every observed word we count which words appear to its left/right
    at distances 1..max_n. Totals are cached to avoid re-summing.
    Autocomplete uses left side only; right side kept for compatibility
    and for optional diffusion-style generation.
    """

    def __init__(self, max_n: int = 3) -> None:
        if max_n < 1:
            raise ValueError(f"max_n must be >= 1, got {max_n}")
        self.max_n = max_n
        self.left_counts: dict[int, dict[tuple[str, ...], Counter[str]]] = {
            n: defaultdict(Counter) for n in range(1, max_n + 1)
        }
        self.right_counts: dict[int, dict[tuple[str, ...], Counter[str]]] = {
            n: defaultdict(Counter) for n in range(1, max_n + 1)
        }
        self.left_totals: dict[int, dict[tuple[str, ...], int]] = {
            n: {} for n in range(1, max_n + 1)
        }
        self.right_totals: dict[int, dict[tuple[str, ...], int]] = {
            n: {} for n in range(1, max_n + 1)
        }
        self.unigrams: Counter[str] = Counter()
        self.vocab: set[str] = set()
        self.sentences: int = 0
        self.tokens: int = 0

    @classmethod
    def from_text(cls, text: str, max_n: int = 3) -> "BidirectionalTopology":
        topo = cls(max_n=max_n)
        topo.ingest(text)
        return topo

    def ingest(self, text: str) -> None:
        if not text or not text.strip():
            raise ValueError("corpus text is empty")
        flattened = WHITESPACE_RE.sub(" ", text).strip()
        ingested = 0
        for sentence in SENTENCE_SPLIT_RE.split(flattened):
            words = tokenize(sentence)
            if not words:
                continue
            self.sentences += 1
            ingested += len(words)
            self.vocab.update(words)
            self.unigrams.update(words)
            for n in range(1, self.max_n + 1):
                for i, target in enumerate(words):
                    if i >= n:
                        ctx = tuple(words[i - n: i])
                        self.left_counts[n][ctx][target] += 1
                        self.left_totals[n][ctx] = self.left_totals[n].get(ctx, 0) + 1
                    if i + n < len(words):
                        ctx = tuple(words[i + 1: i + 1 + n])
                        self.right_counts[n][ctx][target] += 1
                        self.right_totals[n][ctx] = self.right_totals[n].get(ctx, 0) + 1
        self.tokens += ingested
        if ingested == 0:
            raise ValueError("corpus text contains no usable tokens")


# ───────────────────────────────────────────────────────────── diffusion engine (legacy) + autocomplete

_FLOOR = 1e-5
_UNIGRAM_WEIGHT = 0.1
_BACKOFF_SIZE = 50

StepCallback = Callable[[list[str], int, int], None]

@dataclass
class DenoiseResult:
    sequence: list[str]
    confidences: list[float]

@dataclass
class CompleteResult:
    prefix: list[str]
    continuation: list[str]
    confidences: list[float]
    full_sequence: list[str]


class DiscreteDiffusionEngine:
    """N-gram engine: diffusion (bidirectional, legacy) + causal autocomplete."""

    def __init__(self, topo: BidirectionalTopology, rng: random.Random | None = None) -> None:
        self.topo = topo
        self.rng = rng if rng is not None else random.Random()
        self._backoff = [w for w, _ in topo.unigrams.most_common(_BACKOFF_SIZE)]

    def _active_contexts(self, seq: Sequence[str], idx: int):
        for n in range(1, self.topo.max_n + 1):
            if idx >= n:
                ctx = tuple(seq[idx - n: idx])
                if MASK not in ctx:
                    yield "left", n, ctx
            if idx + n < len(seq):
                ctx = tuple(seq[idx + 1: idx + 1 + n])
                if MASK not in ctx:
                    yield "right", n, ctx

    def _causal_contexts(self, seq: Sequence[str], idx: int):
        """Only left contexts — for autocomplete."""
        for n in range(1, self.topo.max_n + 1):
            if idx >= n:
                ctx = tuple(seq[idx - n: idx])
                if MASK not in ctx:
                    yield "left", n, ctx

    def candidate_distribution(self, seq: Sequence[str], idx: int) -> dict[str, float]:
        """Pure (no mutation) softmax over candidates for position *idx* — bidirectional (legacy)."""
        base = 0.0
        contrib: dict[str, float] = {}
        for side, n, ctx in self._active_contexts(seq, idx):
            if side == "left":
                counts = self.topo.left_counts[n].get(ctx)
                total = self.topo.left_totals[n].get(ctx, 0)
            else:
                counts = self.topo.right_counts[n].get(ctx)
                total = self.topo.right_totals[n].get(ctx, 0)
            if not counts or total <= 0:
                continue
            base += math.log(_FLOOR) * n
            floor_adj = math.log(_FLOOR) * n
            for word, count in counts.items():
                contrib[word] = contrib.get(word, 0.0) + (
                    math.log(count / total + _FLOOR) * n - floor_adj
                )
        candidates = sorted(set(contrib) | set(self._backoff))
        energies = {
            w: math.log(self.topo.unigrams.get(w, 1) + 1) * _UNIGRAM_WEIGHT + base + contrib.get(w, 0.0)
            for w in candidates
        }
        max_e = max(energies.values())
        exps = {w: math.exp(e - max_e) for w, e in energies.items()}
        total = sum(exps.values())
        return {w: e / total for w, e in exps.items()}

    def causal_distribution(self, seq: Sequence[str], idx: int) -> dict[str, float]:
        """Causal (left-only) distribution — used for autocomplete."""
        base = 0.0
        contrib: dict[str, float] = {}
        for _, n, ctx in self._causal_contexts(seq, idx):
            counts = self.topo.left_counts[n].get(ctx)
            total = self.topo.left_totals[n].get(ctx, 0)
            if not counts or total <= 0:
                continue
            base += math.log(_FLOOR) * n
            floor_adj = math.log(_FLOOR) * n
            for word, count in counts.items():
                contrib[word] = contrib.get(word, 0.0) + (
                    math.log(count / total + _FLOOR) * n - floor_adj
                )
        candidates = sorted(set(contrib) | set(self._backoff))
        energies = {
            w: math.log(self.topo.unigrams.get(w, 1) + 1) * _UNIGRAM_WEIGHT + base + contrib.get(w, 0.0)
            for w in candidates
        }
        max_e = max(energies.values())
        exps = {w: math.exp(e - max_e) for w, e in energies.items()}
        total = sum(exps.values())
        return {w: e / total for w, e in exps.items()}

    def complete(
        self,
        prefix: Sequence[str],
        max_tokens: int = 12,
        temperature: float = 0.35,
        threshold: float = 0.0,
    ) -> CompleteResult:
        """Left-to-right autocomplete: continue prefix token by token."""
        seq = [w.lower() for w in prefix]
        confidences: list[float] = []
        continuation: list[str] = []
        for _ in range(max_tokens):
            idx = len(seq)
            probs = self.causal_distribution(seq, idx)
            if not probs:
                break
            # temperature-scaled sampling
            if temperature <= 0:
                chosen = max(probs, key=probs.get)
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
            # stop early on sentence end if we already generated a few tokens
            if chosen in {".", "!", "?"} and len(continuation) >= 4:
                break
        full_conf = [1.0]*len(prefix) + confidences
        return CompleteResult(prefix=list(prefix), continuation=continuation, confidences=confidences, full_sequence=seq)

    def denoise(
        self,
        target_len: int,
        steps: int,
        prompt: Sequence[str] = (),
        on_step: StepCallback | None = None,
    ) -> DenoiseResult:
        if target_len < 1:
            raise ValueError(f"target_len must be >= 1, got {target_len}")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")

        seq = [MASK] * target_len
        locked: set[int] = set()
        for i, w in enumerate(prompt[:target_len]):
            seq[i] = w.lower()
            locked.add(i)

        confidences = [0.0] * target_len
        for t in range(1, steps + 1):
            temp = 1.2 * (1.0 - t / steps) + 0.2
            current: dict[int, float] = {}
            for i in range(target_len):
                if i in locked or seq[i] != MASK:
                    continue
                probs = self.candidate_distribution(seq, i)
                words = list(probs)
                weights = [p ** (1.0 / temp) for p in probs.values()]
                chosen = self.rng.choices(words, weights=weights, k=1)[0]
                seq[i] = chosen
                current[i] = probs[chosen]
                confidences[i] = probs[chosen]
            re_mask_ratio = max(0.0, 1.0 - t / steps)
            num_to_remask = int(len(current) * re_mask_ratio)
            if num_to_remask > 0 and current:
                for idx in sorted(current, key=current.get)[:num_to_remask]:
                    seq[idx] = MASK
                    confidences[idx] = 0.0
            if on_step is not None:
                on_step(list(seq), t, steps)
        return DenoiseResult(sequence=seq, confidences=confidences)


# ───────────────────────────────────────────────────────────── CLI helpers

logger = logging.getLogger("mllm52")
BANNER = r"""
 ███╗   ███╗██╗     ██╗     ███╗   ███╗      ███████╗   ██╗  ██╗
 ████╗ ████║██║     ██║     ████╗ ████║      ██╔════╝   ██║  ██║
 ██╔████╔██║██║     ██║     ██╔████╔██║█████╗███████╗   ███████║
 ██║╚██╔╝██║██║     ██║     ██║╚██╔╝██║╚════╝╚════██║   ╚════██║
 ██║ ╚═╝ ██║███████╗███████╗██║ ╚═╝ ██║      ███████║██╗██║  ██║
 ╚═╝     ╚═╝╚══════╝╚══════╝╚═╝     ╚═╝      ╚══════╝╚═╝╚═╝  ╚═╝
        Autocomplete · Causal · Ghost-Text · Document Editor
"""

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="MLLM-5.2",
        description="Document autocomplete LM — continues your prefix left-to-right using causal n-gram diffusion topology.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python MLLM-5.2.py "the quick brown" --steps 12 --seed 42
              python MLLM-5.2.py autocomplete "what is an atom" --steps 16
              python MLLM-5.2.py generate "hello world" --show-steps --extra-tokens 8 14
              python MLLM-5.2.py --corpus ./my.txt autocomplete "hello"

            tips:
              --steps is the effort/length knob.
              --temperature low (0.2) = deterministic ghost; high (1.0) = creative.
              --seed makes output reproducible.
              In REPL, Tab accepts ghost, Esc dismisses. Try index.html playground!
        """),
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--corpus", type=Path, default=None, help="path to training corpus (default: embedded)")
    p.add_argument("--seed", type=int, default=None, help="seed for reproducible sampling")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    def add_decoding_args(sub):
        sub.add_argument("--steps", type=int, default=16, help="max tokens to generate / diffusion steps (default: 16)")
        sub.add_argument("--extra-tokens", type=int, nargs=2, default=None, metavar=("MIN", "MAX"),
                         help="range of tokens beyond prefix (default: steps..steps, alias for --steps)")
        sub.add_argument("--temperature", type=float, default=0.35, help="sampling temperature 0.0=greedy ..1.2=creative (default 0.35)")
        sub.add_argument("--threshold", type=float, default=0.0, help="min confidence to emit token (default 0.0)")
        sub.add_argument("--max-ngram", type=int, default=3, help="max n-gram order (default 3)")
        sub.add_argument("--show-steps", action="store_true", help="render intermediate diffusion states (generate only)")
        # allow global flags also after subcommand
        sub.add_argument("--seed", type=int, default=None, dest="seed_sub", help=argparse.SUPPRESS)
        sub.add_argument("--corpus", type=Path, default=None, dest="corpus_sub", help=argparse.SUPPRESS)
        sub.add_argument("--no-color", action="store_true", dest="no_color_sub", help=argparse.SUPPRESS)
        sub.add_argument("-v", "--verbose", action="store_true", dest="verbose_sub", help=argparse.SUPPRESS)

    subs = p.add_subparsers(dest="command")
    ac = subs.add_parser("autocomplete", help="autocomplete a prefix (default)")
    ac.add_argument("prompt", nargs="?", default=None, help="prefix text to continue")
    add_decoding_args(ac)

    gen = subs.add_parser("generate", help="alias for autocomplete (legacy diffusion)")
    gen.add_argument("prompt", nargs="?", default=None, help="prefix text")
    add_decoding_args(gen)

    chat = subs.add_parser("chat", help="interactive autocomplete REPL (alias)")
    add_decoding_args(chat)

    comp = subs.add_parser("complete", help=argparse.SUPPRESS)
    comp.add_argument("prompt", nargs="?", default=None, help=argparse.SUPPRESS)
    add_decoding_args(comp)

    return p


def load_topology(corpus_path: Path | None, term: Term, max_n: int = 3) -> BidirectionalTopology:
    t0 = time.time()
    if corpus_path is not None:
        logger.info("loading corpus from %s", corpus_path)
        text = corpus_path.read_text(encoding="utf-8")
        topo = BidirectionalTopology.from_text(text, max_n=max_n)
        src = str(corpus_path)
    else:
        logger.info("using embedded built-in corpus")
        text = BUILT_IN_CORPUS
        topo = BidirectionalTopology.from_text(text, max_n=max_n)
        src = "embedded"
    dt = time.time() - t0
    print(term.paint("dim", f"· corpus: {src}"))
    print(term.paint("dim", f"· topology: {len(topo.vocab)} vocab · {topo.tokens} tokens · {topo.sentences} sentences · {dt:.2f}s  (causal n={max_n})"))
    logger.info("topology: %d vocab %d tokens %d sents in %.2fs", len(topo.vocab), topo.tokens, topo.sentences, dt)
    return topo


def make_step_printer(term: Term):
    def _print(seq: list[str], step: int, total: int) -> None:
        interval = max(1, total // 5)
        if step % interval != 0 and step != total:
            return
        bar_len = 20
        filled = int(bar_len * (step / total))
        bar = "█" * filled + "░" * (bar_len - filled)
        display = "".join(term.paint("red", "_") if w == MASK else term.paint("green", w[0]) for w in seq)
        print(term.paint("dim", f"[Step {step:02d}/{total}] {bar}") + " " + display)
    return _print


def assemble_autocomplete(prefix_words: Sequence[str], cont: Sequence[str]) -> tuple[str, str]:
    prefix_str = " ".join(prefix_words)
    cont_str = " ".join(cont)
    for mark in (",", ".", "!", "?"):
        cont_str = cont_str.replace(f" {mark}", mark)
    if prefix_str and cont_str and not prefix_str.endswith((" ", ".", "!", "?")):
        prefix_str += " "
    if cont_str:
        # capitalize first char of continuation if prefix ends with sentence boundary
        if not prefix_str or prefix_str.rstrip().endswith((".", "!", "?")):
            cont_str = cont_str[0].upper() + cont_str[1:] if cont_str else cont_str
    return prefix_str, cont_str


def assemble_text(prompt_words: Sequence[str], result: DenoiseResult) -> tuple[str, str]:
    prompt_str = " ".join(prompt_words)
    gen_words = [w for i, w in enumerate(result.sequence) if i >= len(prompt_words) and w != MASK]
    gen_str = " ".join(gen_words)
    for mark in (",", ".", "!", "?"):
        gen_str = gen_str.replace(f" {mark}", mark)
    if prompt_str and not prompt_str.endswith((" ", ".", "!", "?")):
        prompt_str += " "
    if gen_str:
        gen_str = gen_str[0].upper() + gen_str[1:]
    return prompt_str, gen_str


def render_autocomplete(term: Term, prefix_words: Sequence[str], result: CompleteResult, show_heatmap: bool = True) -> None:
    prefix_str, cont_str = assemble_autocomplete(prefix_words, result.continuation)
    print()
    print(term.paint("cyan", term.paint("bold", "Autocomplete:")))
    # ghost style: prefix normal, continuation dim/italic-like
    print(term.paint("cyan", prefix_str) + term.paint("dim", cont_str))
    if show_heatmap and result.continuation:
        print()
        print(term.paint("dim", "Confidence:"))
        chips = []
        for c in result.confidences:
            color = "green" if c > 0.35 else "yellow" if c > 0.15 else "red"
            chips.append(term.paint(color, "█"))
        print("".join(chips) + term.paint("dim", f"  avg {sum(result.confidences)/len(result.confidences):.2f}"))
        print(term.paint("dim", "[Green=High  Yellow=Med  Red=Low]  Tab=accept  Esc=dismiss"))


def render_result(term: Term, prompt_words: Sequence[str], result: DenoiseResult) -> None:
    prompt_str, gen_str = assemble_text(prompt_words, result)
    print()
    print(term.paint("cyan", term.paint("bold", "Full Sculpted Text:")))
    print(term.paint("cyan", prompt_str) + term.paint("magenta", term.paint("bold", gen_str)))
    heat = []
    for i, word in enumerate(result.sequence):
        if word == MASK:
            continue
        if i < len(prompt_words):
            heat.append(term.paint("cyan", "█"))
        else:
            c = result.confidences[i]
            color = "green" if c > 0.6 else "yellow" if c > 0.3 else "red"
            heat.append(term.paint(color, "█"))
    print()
    print(term.paint("dim", "Confidence Heatmap (Prompt | Generated):"))
    print("".join(heat))
    print(term.paint("dim", "[Green=High  Yellow=Med  Red=Low]"))


def _get_max_tokens(args) -> int:
    if getattr(args, "extra_tokens", None):
        lo, hi = args.extra_tokens
        # for autocomplete, use hi as max
        return int(hi)
    return int(getattr(args, "steps", 16))


def do_autocomplete(engine: DiscreteDiffusionEngine, text: str, max_tokens: int, temperature: float, threshold: float) -> tuple[list[str], CompleteResult]:
    prefix_words = tokenize(text) if text else []
    res = engine.complete(prefix=prefix_words, max_tokens=max_tokens, temperature=temperature, threshold=threshold)
    return prefix_words, res


def cmd_autocomplete(args: argparse.Namespace, engine: DiscreteDiffusionEngine, term: Term) -> int:
    # prompt may be in args.prompt or global prefix
    prompt_text = getattr(args, "prompt", None) or getattr(args, "prefix", None) or ""
    if not prompt_text and not sys.stdin.isatty():
        try:
            _stdin = sys.stdin.read().strip()
        except Exception:
            _stdin = ""
        if _stdin:
            prompt_text = _stdin
    if not prompt_text:
        # if no prompt provided via CLI, fall back to REPL
        return cmd_chat(args, engine, term)
    max_tokens = _get_max_tokens(args)
    prefix_words, result = do_autocomplete(engine, prompt_text, max_tokens, float(getattr(args, "temperature", 0.35)), float(getattr(args, "threshold", 0.0)))
    render_autocomplete(term, prefix_words, result)
    return 0


def cmd_generate(args: argparse.Namespace, engine: DiscreteDiffusionEngine, term: Term) -> int:
    # legacy diffusion path if --show-steps requested, else use autocomplete
    if getattr(args, "show_steps", False):
        on_step = make_step_printer(term)
        prompt_text = getattr(args, "prompt", None) or getattr(args, "prefix", None) or ""
        if not prompt_text and not sys.stdin.isatty():
            try:
                _stdin = sys.stdin.read().strip()
            except Exception:
                _stdin = ""
            if _stdin:
                prompt_text = _stdin
        prompt_words = tokenize(prompt_text) if prompt_text else []
        lo, hi = tuple(args.extra_tokens) if args.extra_tokens else (int(args.steps), int(args.steps))
        # for legacy we keep diffusion: use hi as extra
        target_len = len(prompt_words) + engine.rng.randint(lo, hi)
        result = engine.denoise(target_len=target_len, steps=int(args.steps), prompt=prompt_words, on_step=on_step)
        render_result(term, prompt_words, result)
        return 0
    # otherwise autocomplete
    return cmd_autocomplete(args, engine, term)


def cmd_chat(args: argparse.Namespace, engine: DiscreteDiffusionEngine, term: Term) -> int:
    print(term.paint("magenta", term.paint("bold", "\n› MLLM-5.2 Autocomplete — type a prefix, Tab to accept ghost.  :help  [quit] to exit.")))
    print(term.paint("dim", "  (causal n-gram, left context only — continues your document)\n"))
    while True:
        try:
            raw = input(term.paint("bold", "› ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        low = raw.lower()
        if low in {"[quit]", "quit", "exit", ":quit", ":q"}:
            break
        if low in {":help", "help", "?"}:
            print(term.paint("dim", "  commands: [quit]/exit  :help  :clear  :steps N  :temp N  :seed N  :show on|off"))
            print(term.paint("dim", "  flags: --steps N --temperature T --threshold T --max-ngram N --seed N --corpus PATH"))
            print(term.paint("dim", "  playground: open index.html in browser for ghost-text editor"))
            continue
        if low in {":clear", "clear"}:
            os.system("clear" if os.name != "nt" else "cls")
            continue
        if low.startswith(":steps"):
            try:
                n = int(low.split()[1])
                if n < 1: raise ValueError
                args.steps = n
                print(term.paint("dim", f"  steps → {n}"))
            except Exception:
                print(term.paint("red", "  usage: :steps <positive int>"))
            continue
        if low.startswith(":temp"):
            try:
                t = float(low.split()[1])
                args.temperature = t
                print(term.paint("dim", f"  temperature → {t}"))
            except Exception:
                print(term.paint("red", "  usage: :temp <float 0.0-1.2>"))
            continue
        if low.startswith(":seed"):
            try:
                if low.split()[1].lower() == "none":
                    engine.rng = random.Random()
                    print(term.paint("dim", "  seed → random"))
                else:
                    s = int(low.split()[1])
                    engine.rng = random.Random(s)
                    print(term.paint("dim", f"  seed → {s}"))
            except Exception:
                print(term.paint("red", "  usage: :seed <int>  or  :seed none"))
            continue

        # treat input as prefix to autocomplete
        max_tokens = _get_max_tokens(args)
        prefix_words, result = do_autocomplete(engine, raw, max_tokens, float(getattr(args, "temperature", 0.35)), float(getattr(args, "threshold", 0.0)))
        render_autocomplete(term, prefix_words, result)
        print()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    # ── Bare-prefix compatibility: support `python MLLM-5.2.py "hello world" --steps 4`
    # without requiring explicit subcommand, like ghost-text one-shot.
    # This keeps 5.1 architecture (BidirectionalTopology, DiscreteDiffusionEngine, etc.)
    # intact while making one-shot intuitive. If no known subcommand is present,
    # parse flags loosely and treat remaining as prompt (or stdin pipe).
    if argv is None:
        _argv_list = sys.argv[1:]
    else:
        _argv_list = list(argv)
    known_subs = {"autocomplete", "generate", "chat", "complete"}
    has_sub = any(tok in known_subs for tok in _argv_list)
    has_help = any(tok in ("-h", "--help", "--version") for tok in _argv_list)
    if not has_sub and not has_help:
        _tmp = argparse.ArgumentParser(add_help=False)
        _tmp.add_argument("--corpus", type=Path, default=None)
        _tmp.add_argument("--seed", type=int, default=None)
        _tmp.add_argument("--no-color", action="store_true")
        _tmp.add_argument("-v", "--verbose", action="store_true")
        _tmp.add_argument("--steps", type=int, default=16)
        _tmp.add_argument("--extra-tokens", type=int, nargs=2, default=None, metavar=("MIN", "MAX"))
        _tmp.add_argument("--temperature", type=float, default=0.35)
        _tmp.add_argument("--threshold", type=float, default=0.0)
        _tmp.add_argument("--max-ngram", type=int, default=3)
        _tmp.add_argument("--show-steps", action="store_true")
        try:
            _tmp_args, _remaining = _tmp.parse_known_args(_argv_list)
        except SystemExit:
            _tmp_args = None
            _remaining = None  # type: ignore
        if _tmp_args is not None and _remaining is not None:
            _unknown = [t for t in _remaining if t.startswith("-")]
            if not _unknown:
                _prompt_text = " ".join(_remaining).strip()
                _stdin_text = ""
                if not _prompt_text and not sys.stdin.isatty():
                    try:
                        _stdin_text = sys.stdin.read().strip()
                    except Exception:
                        _stdin_text = ""
                    if _stdin_text:
                        _prompt_text = _stdin_text
                # Only take bare path if there's a prompt or stdin or no args (REPL)
                # If _remaining is empty and no stdin, this is a bare REPL invocation -> handle here too
                # Distinguish from unknown-flag case which we already excluded
                # Proceed with bare handling (autocomplete or REPL)
                _seed = _tmp_args.seed
                _corpus_arg = _tmp_args.corpus
                _no_color = bool(_tmp_args.no_color)
                _verbose = bool(_tmp_args.verbose)
                logging.basicConfig(level=logging.DEBUG if _verbose else logging.WARNING,
                                    format="%(levelname)s %(name)s: %(message)s")
                _term = Term.detect(_no_color)
                if _term.enabled:
                    print(_term.paint("cyan", BANNER.strip("\n")))
                    print(_term.paint("dim", f"  v{__version__}  ·  autocomplete LM  ·  causal n={_tmp_args.max_ngram}  ·  ghost-text"))
                    print()
                else:
                    print(f"MLLM-5.2 v{__version__} — autocomplete LM")
                _extra = tuple(_tmp_args.extra_tokens) if _tmp_args.extra_tokens else None
                if _extra and len(_extra) == 2 and _extra[0] > _extra[1]:
                    print("mllm52: error: --extra-tokens MIN must be <= MAX", file=sys.stderr)
                    return 2
                _max_n = int(_tmp_args.max_ngram)
                _corpus_path = Path(_corpus_arg) if _corpus_arg else None
                try:
                    _topo = load_topology(_corpus_path, _term, max_n=_max_n)
                except (OSError, ValueError) as exc:
                    print(f"mllm52: error: cannot load corpus {_corpus_path or 'embedded'}: {exc}", file=sys.stderr)
                    return 2
                if _extra:
                    _tmp_args.extra_tokens = _extra
                else:
                    _tmp_args.extra_tokens = (int(_tmp_args.steps), int(_tmp_args.steps))
                _engine = DiscreteDiffusionEngine(_topo, rng=random.Random(_seed))
                _pseudo = argparse.Namespace(
                    prompt=_prompt_text if _prompt_text else None,
                    prefix=None,
                    steps=_tmp_args.steps,
                    extra_tokens=_tmp_args.extra_tokens,
                    temperature=_tmp_args.temperature,
                    threshold=_tmp_args.threshold,
                    max_ngram=_tmp_args.max_ngram,
                    show_steps=_tmp_args.show_steps,
                    seed=_seed,
                    corpus=_corpus_arg,
                    no_color=_no_color,
                    verbose=_verbose,
                    command="autocomplete" if _prompt_text else None,
                )
                if _prompt_text:
                    if _tmp_args.show_steps:
                        return cmd_generate(_pseudo, _engine, _term)
                    return cmd_autocomplete(_pseudo, _engine, _term)
                else:
                    # No prompt and no stdin -> REPL, but only if not already handled as help
                    # If argv_list was empty, this is plain REPL invocation
                    # Also if argv_list contained only flags, go to REPL with those flags
                    return cmd_chat(_pseudo, _engine, _term)
    parser = build_parser()
    args = parser.parse_args(argv)
    # allow global flags also after subcommand
    seed = getattr(args, "seed_sub", None) if getattr(args, "seed_sub", None) is not None else getattr(args, "seed", None)
    corpus_arg = getattr(args, "corpus_sub", None) if getattr(args, "corpus_sub", None) is not None else getattr(args, "corpus", None)
    no_color = bool(getattr(args, "no_color", False) or getattr(args, "no_color_sub", False))
    verbose = bool(getattr(args, "verbose", False) or getattr(args, "verbose_sub", False))

    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    term = Term.detect(no_color)

    if term.enabled:
        print(term.paint("cyan", BANNER.strip("\n")))
        print(term.paint("dim", f"  v{__version__}  ·  autocomplete LM  ·  causal n=3  ·  ghost-text"))
        print()
    else:
        print(f"MLLM-5.2 v{__version__} — autocomplete LM")



    # validate extra-tokens
    try:
        extra = tuple(args.extra_tokens) if getattr(args, "extra_tokens", None) else None
    except Exception:
        extra = None
    if extra and len(extra) == 2 and extra[0] > extra[1]:
        print("mllm52: error: --extra-tokens MIN must be <= MAX", file=sys.stderr)
        return 2

    max_n = int(getattr(args, "max_ngram", 3)) if hasattr(args, "max_ngram") else 3
    corpus_path = Path(corpus_arg) if corpus_arg else None
    try:
        topo = load_topology(corpus_path, term, max_n=max_n)
    except (OSError, ValueError) as exc:
        print(f"mllm52: error: cannot load corpus {corpus_path or 'embedded'}: {exc}", file=sys.stderr)
        return 2

    # patch args for downstream
    if extra:
        args.extra_tokens = extra
    else:
        # default extra_tokens = steps..steps for autocomplete
        args.extra_tokens = (int(getattr(args, "steps", 16)), int(getattr(args, "steps", 16)))
    if not hasattr(args, "steps") or args.steps is None:
        args.steps = 16
    if not hasattr(args, "temperature"):
        args.temperature = 0.35
    if not hasattr(args, "threshold"):
        args.threshold = 0.0

    engine = DiscreteDiffusionEngine(topo, rng=random.Random(seed))
    cmd = getattr(args, "command", None)
    if cmd in ("autocomplete", "complete"):
        # Let cmd_autocomplete handle prompt or stdin or fallback to REPL
        return cmd_autocomplete(args, engine, term)
    if cmd is None and getattr(args, "prefix", None) is None:
        # No subcommand and no prefix attribute -> default REPL (or bare already handled)
        # Check stdin for one-shot without subcommand (should have been handled earlier, but keep)
        if not sys.stdin.isatty():
            try:
                _stdin_prompt = sys.stdin.read().strip()
            except Exception:
                _stdin_prompt = ""
            if _stdin_prompt:
                _pseudo = argparse.Namespace(prompt=_stdin_prompt, prefix=None, steps=getattr(args, "steps", 16),
                                             extra_tokens=getattr(args, "extra_tokens", None),
                                             temperature=getattr(args, "temperature", 0.35),
                                             threshold=getattr(args, "threshold", 0.0),
                                             max_ngram=getattr(args, "max_ngram", 3),
                                             show_steps=getattr(args, "show_steps", False))
                return cmd_autocomplete(_pseudo, engine, term)
        return cmd_chat(args, engine, term)
    if cmd == "generate":
        return cmd_generate(args, engine, term)
    if cmd == "chat":
        return cmd_chat(args, engine, term)
    return cmd_chat(args, engine, term)


if __name__ == "__main__":
    raise SystemExit(main())
