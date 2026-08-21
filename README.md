# MLLM-5.2

> **Preview** — APIs and behavior may change before stable. Preview artifact.

Lightweight autocomplete micro language models — single-file, zero-dependency, local-first.

Document Autocomplete that continues your document left-to-right. Type a prefix → dim ghost text inline → `Tab` to accept, `Esc` to dismiss. **Input → Ghost → Output.** No server, no install.

> Single-file, zero-dep. `python MLLM-5.2-Preview-0P.py` is the whole model — same engine as 5.1, just input and output.

## What's new in 5.2 vs 5.1

Architecture unchanged from 5.1 — same topology and engine (`CausalTopology` + `AutocompleteEngine`).
Corpus is autocomplete-based — ghost-text pairs (prefix → continuation) tuned for Tab-to-accept quality. Lightweight preview, pure Python 3.10+ stdlib.

## Quick start

Requires Python 3.10+, no `pip`.

```bash
# one-shot autocomplete
python MLLM-5.2-Preview-0P.py "hello world" --steps 12 --seed 42
python MLLM-5.2-Preview-0P.py autocomplete "hello world" --steps 16 --seed 7

# interactive REPL
python MLLM-5.2-Preview-0P.py
# :help, :clear, :steps N, :temp N, :seed N, [quit] to exit

# pipe / stdin
echo "hello world" | python MLLM-5.2-Preview-0P.py --steps 10 --plain
```

`--plain` prints `prefix + continuation` for pipes. Omit for ANSI ghost view with confidence heatmap.

## Corpus — place to define

`MLLM-5.2-Preview-0P.py` is a place to define your corpus. Edit the `BUILT_IN_CORPUS` placeholder at the top of the file and paste your autocomplete-based examples (prefix → continuation).

- Format: UTF-8 text, one example per line ideal; paragraphs also work.
- Tokenization: `\b[a-zA-Z0-9']+\b|[.!?]` lowercased, split on `(?<=[.!?])\s+`.
- Sweet spot: 5KB–500KB of examples for immediate ghost quality.
- Optional override: `--corpus path/to/file.txt` for a one-off external file (not primary).

## Flags

| Flag | Default | Description |
|---|---|---|
| `--steps N` | `16` | Max tokens to generate |
| `--temperature T` | `0.35` | 0.0 greedy → 1.2 creative |
| `--threshold T` | `0.0` | Min confidence to emit |
| `--max-ngram N` | `3` | Context size (1–5) |
| `--seed N` | random | Deterministic sampling |
| `--corpus PATH` | embedded | One-off external corpus override |
| `--no-color` / `--plain` | auto/off | Disable ANSI / plain pipe output |

Subcommands `autocomplete` / `generate` / `complete` / `chat` share flags. Bare prefix also works.

## How it works

1. Topology — tokenize and build causal `left_counts[n][ctx][word]` for n=1..3.
2. Score — `log(count/total)*n` summed + unigram backoff.
3. Sample — `softmax → p^(1/T)` with `random.Random(seed)`; stop on `p < threshold` or `.!?` after ≥4 tokens.

## Layout

```
MLLM-5.2-Preview-0P.py  ← self-contained runner — zero-dep, input → output, BUILT_IN_CORPUS inside
README.md               ← this file
LICENSE                 ← GPL
```

## Coming soon

Official release with 5 sized models — `0P` (no corpus, bring your own), `1P`, `5P`, `50P`, `100P` — same engine, different corpus scales. Lightweight autocomplete remains tiny and local.
Document Editor — single-file editor built around 5.2 ghost-text: inline ghost, Tab / Esc, confidence heatmap, temperature and length controls, local and instant.
---

*Core: causal n-gram, autocomplete ghost-text, confidence-gated, deterministic with --seed, lightweight.*
