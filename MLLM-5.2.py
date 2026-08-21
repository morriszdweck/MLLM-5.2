#!/usr/bin/env python3
"""
MLLM-5.2 — Lightweight Autocomplete Micro Language Model
BYO corpus / Input -> Ghost -> Output

Single-file, zero-dependency, pure Python 3.10+
Causal n-gram autocomplete (left context n=1..3), temperature sampling.

Usage:
  python MLLM-5.2.py --corpus corpus.txt "the quick brown"   -> prints continuation
  python MLLM-5.2.py --corpus corpus.txt                      -> REPL: type prefix, get continuation
  echo "hello world" | python MLLM-5.2.py --corpus corpus.txt --steps 10 --temperature 0.3
  python MLLM-5.2.py --corpus corpus.txt autocomplete "hello world" --steps 16

Requires --corpus PATH. If not provided:
  "No corpus found — provide --corpus your.txt or place text in 'MLLM 5.2 preview' and run --corpus 'MLLM 5.2 preview'"
"""
from __future__ import annotations
import argparse
import math
import os
import random
import re
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

__version__ = "5.2"

TOKEN_RE = re.compile(r"\b[a-zA-Z0-9']+\b|[.!?]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_RE = re.compile(r"\s+")
def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())

_CODES = {"bold": "\033[1m", "dim": "\033[2m", "cyan": "\033[36m", "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m", "magenta": "\033[35m"}
_RESET = "\033[0m"
class Term:
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
class CausalTopology:
    """Causal n-gram statistics (left context only, n=1..max_n)."""
    def __init__(self, max_n: int = 3) -> None:
        if max_n < 1:
            raise ValueError(f"max_n must be >= 1, got {max_n}")
        self.max_n = max_n
        self.left_counts: dict[int, dict[tuple[str, ...], Counter[str]]] = {n: defaultdict(Counter) for n in range(1, max_n + 1)}
        self.left_totals: dict[int, dict[tuple[str, ...], int]] = {n: {} for n in range(1, max_n + 1)}
        self.unigrams: Counter[str] = Counter()
        self.vocab: set[str] = set()
        self.sentences: int = 0
        self.tokens: int = 0
    @classmethod
    def from_text(cls, text: str, max_n: int = 3) -> "CausalTopology":
        t = cls(max_n=max_n)
        t.ingest(text)
        return t
    def ingest(self, text: str) -> None:
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
                        ctx = tuple(words[i - n:i])
                        self.left_counts[n][ctx][tgt] += 1
                        self.left_totals[n][ctx] = self.left_totals[n].get(ctx, 0) + 1
        self.tokens += ingested
        if ingested == 0:
            raise ValueError("corpus text contains no usable tokens")

_FLOOR = 1e-5
_UNIGRAM_WEIGHT = 0.1
_BACKOFF_SIZE = 50
@dataclass
class CompleteResult:
    prefix: list[str]
    continuation: list[str]
    confidences: list[float]
    full_sequence: list[str]
class AutocompleteEngine:
    def __init__(self, topo: CausalTopology, rng: random.Random | None = None) -> None:
        self.topo = topo
        self.rng = rng if rng is not None else random.Random()
        self._backoff = [w for w, _ in topo.unigrams.most_common(_BACKOFF_SIZE)]
    def _causal_contexts(self, seq: Sequence[str], idx: int):
        for n in range(1, self.topo.max_n + 1):
            if idx >= n:
                ctx = tuple(seq[idx - n: idx])
                yield n, ctx
    def causal_distribution(self, seq: Sequence[str], idx: int) -> dict[str, float]:
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
                contrib[word] = contrib.get(word, 0.0) + (math.log(cnt / total + _FLOOR) * n - floor_adj)
        candidates = sorted(set(contrib) | set(self._backoff))
        energies = {w: math.log(self.topo.unigrams.get(w, 1) + 1) * _UNIGRAM_WEIGHT + base + contrib.get(w, 0.0) for w in candidates}
        max_e = max(energies.values())
        exps = {w: math.exp(e - max_e) for w, e in energies.items()}
        total = sum(exps.values())
        return {w: e / total for w, e in exps.items()}
    def complete(self, prefix: Sequence[str], max_tokens: int = 16, temperature: float = 0.35, threshold: float = 0.0) -> CompleteResult:
        seq = [w.lower() for w in prefix]
        confidences: list[float] = []
        continuation: list[str] = []
        for _ in range(max_tokens):
            idx = len(seq)
            probs = self.causal_distribution(seq, idx)
            if not probs:
                break
            if temperature <= 0:
                chosen = max(probs, key=probs.get)  # type: ignore
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
        return CompleteResult(prefix=list(prefix), continuation=continuation, confidences=confidences, full_sequence=seq)
BANNER = "MLLM-5.2 — Lightweight Autocomplete"

CORPUS_ERROR = "No corpus found — provide --corpus your.txt or place text in 'MLLM 5.2 preview' and run --corpus 'MLLM 5.2 preview'"
def assemble_autocomplete(prefix_words: Sequence[str], cont: Sequence[str]) -> tuple[str, str]:
    prefix_str = " ".join(prefix_words)
    cont_str = " ".join(cont)
    for mark in (",", ".", "!", "?"):
        cont_str = cont_str.replace(f" {mark}", mark)
    if prefix_str and cont_str and not prefix_str.endswith((" ", ".", "!", "?")):
        prefix_str += " "
    if prefix_str and cont_str and prefix_str.rstrip().endswith((".", "!", "?")):
        cont_str = cont_str[0].upper() + cont_str[1:] if cont_str else cont_str
    elif not prefix_str and cont_str:
        cont_str = cont_str[0].upper() + cont_str[1:] if cont_str else cont_str
    return prefix_str, cont_str
def render_autocomplete(term: Term, prefix_words: Sequence[str], result: CompleteResult, plain: bool = False) -> None:
    prefix_str, cont_str = assemble_autocomplete(prefix_words, result.continuation)
    if plain:
        print(prefix_str + cont_str)
        return
    print()
    print(term.paint("cyan", term.paint("bold", "Autocomplete:")))
    print(term.paint("cyan", prefix_str) + term.paint("dim", cont_str))
    if result.continuation:
        print()
        print(term.paint("dim", "Confidence:"))
        chips = []
        for c in result.confidences:
            color = "green" if c > 0.35 else "yellow" if c > 0.15 else "red"
            chips.append(term.paint(color, "█"))
        avg = sum(result.confidences) / len(result.confidences) if result.confidences else 0
        print("".join(chips) + term.paint("dim", f"  avg {avg:.2f}"))
        print(term.paint("dim", "[Green=High  Yellow=Med  Red=Low]  Tab=accept  Esc=dismiss"))
def do_autocomplete(engine: AutocompleteEngine, text: str, max_tokens: int, temperature: float, threshold: float):
    prefix_words = tokenize(text) if text else []
    res = engine.complete(prefix=prefix_words, max_tokens=max_tokens, temperature=temperature, threshold=threshold)
    return prefix_words, res
def load_topology(corpus_path: Path, term: Term, max_n: int, plain: bool = False) -> CausalTopology:
    import time
    t0 = time.time()
    text = corpus_path.read_text(encoding="utf-8")
    topo = CausalTopology.from_text(text, max_n=max_n)
    dt = time.time() - t0
    if not plain:
        print(term.paint("dim", f"· corpus: {corpus_path}"))
        print(term.paint("dim", f"· topology: {len(topo.vocab)} vocab · {topo.tokens} tokens · {topo.sentences} sentences · {dt:.2f}s  (causal n={max_n})"))
    return topo
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="MLLM-5.2",
        description="MLLM-5.2 — Lightweight Autocomplete Micro Language Model / BYO corpus / Input -> Ghost -> Output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python MLLM-5.2.py --corpus corpus.txt "the quick brown"
              python MLLM-5.2.py --corpus corpus.txt --steps 10 --temperature 0.3 "hello world"
              python MLLM-5.2.py --corpus corpus.txt autocomplete "hello world" --steps 16
              echo "hello world" | python MLLM-5.2.py --corpus corpus.txt --steps 10
              python MLLM-5.2.py --corpus corpus.txt   # REPL
        """),
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--corpus", type=Path, default=None, required=False, help="path to training corpus (required)")
    p.add_argument("--steps", type=int, default=16, help="max tokens to generate (default: 16)")
    p.add_argument("--temperature", type=float, default=0.35, help="sampling temperature 0.0=greedy ..1.2=creative (default 0.35)")
    p.add_argument("--threshold", type=float, default=0.0, help="min confidence to emit token (default 0.0)")
    p.add_argument("--max-ngram", type=int, default=3, help="max n-gram order (default 3)")
    p.add_argument("--seed", type=int, default=None, help="seed for reproducible sampling")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    p.add_argument("--plain", action="store_true", help="plain output: just print continuation (for pipes/tests)")
    subs = p.add_subparsers(dest="command")
    ac = subs.add_parser("autocomplete", help="autocomplete a prefix")
    ac.add_argument("prompt", nargs="?", default=None, help="prefix text to continue")
    ac.add_argument("--steps", type=int, default=None, help=argparse.SUPPRESS)
    ac.add_argument("--temperature", type=float, default=None, help=argparse.SUPPRESS)
    ac.add_argument("--threshold", type=float, default=None, help=argparse.SUPPRESS)
    ac.add_argument("--max-ngram", type=int, default=None, help=argparse.SUPPRESS)
    ac.add_argument("--seed", type=int, default=None, dest="seed_sub", help=argparse.SUPPRESS)
    ac.add_argument("--corpus", type=Path, default=None, dest="corpus_sub", help=argparse.SUPPRESS)
    ac.add_argument("--no-color", action="store_true", dest="no_color_sub", help=argparse.SUPPRESS)
    ac.add_argument("--plain", action="store_true", dest="plain_sub", help=argparse.SUPPRESS)
    for alias in ("generate", "complete"):
        sp = subs.add_parser(alias, help=argparse.SUPPRESS)
        sp.add_argument("prompt", nargs="?", default=None, help=argparse.SUPPRESS)
        sp.add_argument("--steps", type=int, default=None, help=argparse.SUPPRESS)
        sp.add_argument("--temperature", type=float, default=None, help=argparse.SUPPRESS)
        sp.add_argument("--threshold", type=float, default=None, help=argparse.SUPPRESS)
        sp.add_argument("--max-ngram", type=int, default=None, help=argparse.SUPPRESS)
        sp.add_argument("--seed", type=int, default=None, dest="seed_sub", help=argparse.SUPPRESS)
        sp.add_argument("--corpus", type=Path, default=None, dest="corpus_sub", help=argparse.SUPPRESS)
        sp.add_argument("--no-color", action="store_true", dest="no_color_sub", help=argparse.SUPPRESS)
        sp.add_argument("--plain", action="store_true", dest="plain_sub", help=argparse.SUPPRESS)
    ch = subs.add_parser("chat", help="interactive REPL")
    ch.add_argument("--steps", type=int, default=None, help=argparse.SUPPRESS)
    ch.add_argument("--temperature", type=float, default=None, help=argparse.SUPPRESS)
    ch.add_argument("--threshold", type=float, default=None, help=argparse.SUPPRESS)
    ch.add_argument("--max-ngram", type=int, default=None, help=argparse.SUPPRESS)
    ch.add_argument("--seed", type=int, default=None, dest="seed_sub", help=argparse.SUPPRESS)
    ch.add_argument("--corpus", type=Path, default=None, dest="corpus_sub", help=argparse.SUPPRESS)
    ch.add_argument("--no-color", action="store_true", dest="no_color_sub", help=argparse.SUPPRESS)
    ch.add_argument("--plain", action="store_true", dest="plain_sub", help=argparse.SUPPRESS)
    return p
def cmd_repl(args: argparse.Namespace, engine: AutocompleteEngine, term: Term) -> int:
    plain = bool(getattr(args, "plain", False) or getattr(args, "plain_sub", False))
    print(term.paint("magenta", term.paint("bold", "\n› MLLM-5.2 Autocomplete — type a prefix, ghost continues.  :help  [quit] to exit.")))
    print(term.paint("dim", "  (causal n-gram left context n=1..3 — BYO corpus)\n"))
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
            print(term.paint("dim", "  commands: [quit]/exit  :help  :clear  :steps N  :temp N  :seed N"))
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
        steps = int(getattr(args, "steps", 16))
        temp = float(getattr(args, "temperature", 0.35))
        thresh = float(getattr(args, "threshold", 0.0))
        prefix_words, result = do_autocomplete(engine, raw, steps, temp, thresh)
        render_autocomplete(term, prefix_words, result, plain=plain)
        print()
    return 0
def main(argv: Sequence[str] | None = None) -> int:
    # Normalize argv
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)

    # Handle --help / --version early for bare mode (avoid corpus error)
    if any(a in ("-h", "--help") for a in argv):
        build_parser().print_help(sys.stdout)
        return 0
    if "--version" in argv:
        print(f"MLLM-5.2 {__version__}")
        return 0

    known_subs = {"autocomplete", "generate", "complete", "chat"}
    has_sub = any(tok in known_subs for tok in argv)
    if has_sub:
        # Subcommand path: use full parser
        parser = build_parser()
        try:
            args = parser.parse_args(argv)
        except SystemExit as e:
            return int(e.code) if e.code is not None else 2
        seed = getattr(args, "seed_sub", None) if getattr(args, "seed_sub", None) is not None else getattr(args, "seed", None)
        corpus_arg = getattr(args, "corpus_sub", None) if getattr(args, "corpus_sub", None) is not None else getattr(args, "corpus", None)
        no_color = bool(getattr(args, "no_color", False) or getattr(args, "no_color_sub", False))
        plain = bool(getattr(args, "plain", False) or getattr(args, "plain_sub", False))
        # sub may override with None, so fallback
        steps = int(getattr(args, "steps", 16) if getattr(args, "steps", None) is not None else 16)
        temperature = float(getattr(args, "temperature", 0.35) if getattr(args, "temperature", None) is not None else 0.35)
        threshold = float(getattr(args, "threshold", 0.0) if getattr(args, "threshold", None) is not None else 0.0)
        _mn = getattr(args, "max_ngram", None)
        max_n = int(_mn if _mn is not None else 3)
        term = Term.detect(no_color or plain)
        if corpus_arg is None:
            print(f"mllm52: error: {CORPUS_ERROR}", file=sys.stderr)
            parser.print_help(sys.stderr)
            return 2
        corpus_path = Path(corpus_arg)
        if not corpus_path.exists():
            if str(corpus_path) == "MLLM 5.2 preview":
                print(f"mllm52: error: {CORPUS_ERROR} (file '{corpus_path}' not found)", file=sys.stderr)
            else:
                print(f"mllm52: error: corpus file not found: {corpus_path}", file=sys.stderr)
                print(f"  hint: {CORPUS_ERROR}", file=sys.stderr)
            return 2
        try:
            topo = load_topology(corpus_path, term, max_n=max_n, plain=plain)
        except (OSError, ValueError) as exc:
            print(f"mllm52: error: cannot load corpus {corpus_path}: {exc}", file=sys.stderr)
            return 2
        if not plain:
            if term.enabled:
                print(term.paint("cyan", BANNER.strip("\n")))
                print(term.paint("dim", f"  v{__version__}  ·  BYO corpus  ·  causal n={max_n}  ·  Input -> Ghost -> Output"))
                print()
            else:
                print(f"MLLM-5.2 v{__version__} — BYO corpus · causal n={max_n}")
        stdin_text = ""
        if not sys.stdin.isatty():
            try:
                stdin_text = sys.stdin.read().strip()
            except Exception:
                stdin_text = ""
        engine = AutocompleteEngine(topo, rng=random.Random(seed))
        cmd = getattr(args, "command", None)
        if cmd == "chat":
            return cmd_repl(args, engine, term)
        # autocomplete / generate / complete
        prompt_text = getattr(args, "prompt", None)
        if prompt_text:
            text = prompt_text
        elif stdin_text:
            text = stdin_text
        else:
            return cmd_repl(args, engine, term)
        prefix_words, result = do_autocomplete(engine, text, steps, temperature, threshold)
        if plain or (not term.enabled and not sys.stdout.isatty()):
            prefix_str, cont_str = assemble_autocomplete(prefix_words, result.continuation)
            print(prefix_str + cont_str)
        else:
            render_autocomplete(term, prefix_words, result, plain=False)
        return 0
    else:
        # Bare-prefix path: no subcommand, handle flags + leftover as prefix
        # Use a lightweight flags-only parser
        flags = argparse.ArgumentParser(add_help=False)
        flags.add_argument("--corpus", type=Path, default=None)
        flags.add_argument("--steps", type=int, default=16)
        flags.add_argument("--temperature", type=float, default=0.35)
        flags.add_argument("--threshold", type=float, default=0.0)
        flags.add_argument("--max-ngram", type=int, default=3)
        flags.add_argument("--seed", type=int, default=None)
        flags.add_argument("--no-color", action="store_true")
        flags.add_argument("--plain", action="store_true")
        # parse known args; remaining is prefix tokens
        try:
            f_args, remaining = flags.parse_known_args(argv)
        except SystemExit as e:
            return int(e.code) if e.code is not None else 2
        # Detect unknown flags (remaining that looks like --flag)
        for tok in remaining:
            if tok.startswith("-") and tok not in ("-", "--"):
                print(f"mllm52: error: unrecognized arguments: {tok}", file=sys.stderr)
                build_parser().print_help(sys.stderr)
                return 2
        # Strip leading -- separator if present
        if remaining and remaining[0] == "--":
            remaining = remaining[1:]
        # Join remaining tokens as prefix (supports both quoted single arg and unquoted multiple)
        prefix_text = " ".join(remaining).strip()
        # Also check for unknown bare prefix that was split by shell: we already joined
        no_color = bool(f_args.no_color)
        plain = bool(f_args.plain)
        term = Term.detect(no_color or plain)
        if f_args.corpus is None:
            print(f"mllm52: error: {CORPUS_ERROR}", file=sys.stderr)
            build_parser().print_help(sys.stderr)
            return 2
        corpus_path = Path(f_args.corpus)
        if not corpus_path.exists():
            if str(corpus_path) == "MLLM 5.2 preview":
                print(f"mllm52: error: {CORPUS_ERROR} (file '{corpus_path}' not found)", file=sys.stderr)
            else:
                print(f"mllm52: error: corpus file not found: {corpus_path}", file=sys.stderr)
                print(f"  hint: {CORPUS_ERROR}", file=sys.stderr)
            return 2
        try:
            topo = load_topology(corpus_path, term, max_n=int(f_args.max_ngram), plain=plain)
        except (OSError, ValueError) as exc:
            print(f"mllm52: error: cannot load corpus {corpus_path}: {exc}", file=sys.stderr)
            return 2
        if not plain:
            if term.enabled:
                print(term.paint("cyan", BANNER.strip("\n")))
                print(term.paint("dim", f"  v{__version__}  ·  BYO corpus  ·  causal n={int(f_args.max_ngram)}  ·  Input -> Ghost -> Output"))
                print()
            else:
                print(f"MLLM-5.2 v{__version__} — BYO corpus · causal n={int(f_args.max_ngram)}")
        stdin_text = ""
        if not sys.stdin.isatty():
            try:
                stdin_text = sys.stdin.read().strip()
            except Exception:
                stdin_text = ""
        engine = AutocompleteEngine(topo, rng=random.Random(f_args.seed))
        steps = int(f_args.steps)
        temperature = float(f_args.temperature)
        threshold = float(f_args.threshold)
        # Create a minimal args namespace for REPL
        repl_args = argparse.Namespace(steps=steps, temperature=temperature, threshold=threshold, plain=plain, no_color=no_color)
        # Determine text source: explicit prefix > stdin > REPL
        if prefix_text:
            text = prefix_text
        elif stdin_text:
            text = stdin_text
        else:
            return cmd_repl(repl_args, engine, term)
        prefix_words, result = do_autocomplete(engine, text, steps, temperature, threshold)
        if plain or (not term.enabled and not sys.stdout.isatty()):
            prefix_str, cont_str = assemble_autocomplete(prefix_words, result.continuation)
            print(prefix_str + cont_str)
        else:
            render_autocomplete(term, prefix_words, result, plain=False)
        return 0
if __name__ == "__main__":
    raise SystemExit(main())
