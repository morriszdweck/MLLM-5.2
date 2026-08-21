"""Command-line interface for mllm52 (lightweight autocomplete)."""

from __future__ import annotations

import argparse
import os
import random
import sys
import textwrap
from pathlib import Path
from typing import Sequence

from . import __version__
from .engine import AutocompleteEngine
from .terminal import Term
from .topology import CausalTopology, tokenize

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


def render_autocomplete(term: Term, prefix_words: Sequence[str], result, plain: bool = False) -> None:
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


def load_topology(corpus_path: Path, term: Term, max_n: int = 3, plain: bool = False) -> CausalTopology:
    """Read *corpus_path* and build the topology.

    Raises:
        SystemExit or prints error and returns error code if file missing.

    The caller may handle FileNotFound explicitly; this helper also validates
    existence and raises a user-friendly error if missing.

    On success returns a CausalTopology.
    """
    import time

    if not corpus_path.exists():
        # Mirror MLLM-5.2.py error messages
        if str(corpus_path) == "MLLM 5.2 preview":
            print(f"mllm52: error: {CORPUS_ERROR} (file '{corpus_path}' not found)", file=sys.stderr)
        else:
            print(f"mllm52: error: corpus file not found: {corpus_path}", file=sys.stderr)
            print(f"  hint: {CORPUS_ERROR}", file=sys.stderr)
        raise SystemExit(2)
    t0 = time.time()
    try:
        text = corpus_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"mllm52: error: cannot load corpus {corpus_path}: {exc}", file=sys.stderr)
        raise SystemExit(2)
    try:
        topo = CausalTopology.from_text(text, max_n=max_n)
    except ValueError as exc:
        print(f"mllm52: error: cannot load corpus {corpus_path}: {exc}", file=sys.stderr)
        raise SystemExit(2)
    dt = time.time() - t0
    if not plain:
        print(term.paint("dim", f"· corpus: {corpus_path}"))
        print(term.paint("dim", f"· topology: {len(topo.vocab)} vocab · {topo.tokens} tokens · {topo.sentences} sentences · {dt:.2f}s  (causal n={max_n})"))
    return topo


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mllm52",
        description="MLLM-5.2 — Lightweight Autocomplete Micro Language Model / BYO corpus / Input -> Ghost -> Output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              mllm52 --corpus corpus.txt "the quick brown"
              mllm52 --corpus corpus.txt --steps 10 --temperature 0.3 "hello world"
              mllm52 --corpus corpus.txt autocomplete "hello world" --steps 16
              echo "hello world" | mllm52 --corpus corpus.txt --steps 10
              mllm52 --corpus corpus.txt   # REPL
              python -m mllm52 --corpus "MLLM 5.2 preview" "hello world"
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
                if n < 1:
                    raise ValueError
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
        parser = build_parser()
        try:
            args = parser.parse_args(argv)
        except SystemExit as e:
            # If --corpus was missing, also show friendly CORPUS_ERROR
            if "--corpus" not in argv and e.code == 2:
                print(f"mllm52: error: {CORPUS_ERROR}", file=sys.stderr)
            return int(e.code) if e.code is not None else 2
        seed = getattr(args, "seed_sub", None) if getattr(args, "seed_sub", None) is not None else getattr(args, "seed", None)
        corpus_arg = getattr(args, "corpus_sub", None) if getattr(args, "corpus_sub", None) is not None else getattr(args, "corpus", None)
        no_color = bool(getattr(args, "no_color", False) or getattr(args, "no_color_sub", False))
        plain = bool(getattr(args, "plain", False) or getattr(args, "plain_sub", False))
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
        try:
            topo = load_topology(corpus_path, term, max_n=max_n, plain=plain)
        except SystemExit as e:
            return int(e.code) if e.code else 2
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
        # Bare-prefix path
        flags = argparse.ArgumentParser(add_help=False)
        flags.add_argument("--corpus", type=Path, default=None, required=False)
        flags.add_argument("--steps", type=int, default=16)
        flags.add_argument("--temperature", type=float, default=0.35)
        flags.add_argument("--threshold", type=float, default=0.0)
        flags.add_argument("--max-ngram", type=int, default=3)
        flags.add_argument("--seed", type=int, default=None)
        flags.add_argument("--no-color", action="store_true")
        flags.add_argument("--plain", action="store_true")
        try:
            f_args, remaining = flags.parse_known_args(argv)
        except SystemExit as e:
            # argparse will have printed error about missing --corpus
            # Re-emit our friendly CORPUS_ERROR for consistency
            if f_args if 'f_args' in locals() else True:
                # Check if corpus missing
                if "--corpus" not in argv:
                    print(f"mllm52: error: {CORPUS_ERROR}", file=sys.stderr)
            return int(e.code) if e.code is not None else 2
        for tok in remaining:
            if tok.startswith("-") and tok not in ("-", "--"):
                print(f"mllm52: error: unrecognized arguments: {tok}", file=sys.stderr)
                build_parser().print_help(sys.stderr)
                return 2
        if remaining and remaining[0] == "--":
            remaining = remaining[1:]
        prefix_text = " ".join(remaining).strip()
        no_color = bool(f_args.no_color)
        plain = bool(f_args.plain)
        term = Term.detect(no_color or plain)
        if f_args.corpus is None:
            print(f"mllm52: error: {CORPUS_ERROR}", file=sys.stderr)
            build_parser().print_help(sys.stderr)
            return 2
        corpus_path = Path(f_args.corpus)
        try:
            topo = load_topology(corpus_path, term, max_n=int(f_args.max_ngram), plain=plain)
        except SystemExit as e:
            return int(e.code) if e.code else 2
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
        repl_args = argparse.Namespace(steps=steps, temperature=temperature, threshold=threshold, plain=plain, no_color=no_color)
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
