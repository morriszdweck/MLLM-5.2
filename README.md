# MLLM-5.2

> **Stable** — Official release. No longer in preview.

> 🎉 **ALL MODELS RELEASED** — including **Golden — 3981P**, the largest autocomplete model yet — single-file, zero-dep, local-first.

Lightweight autocomplete micro language models — single-file, zero-dependency, local-first.

> Single-file, zero-dep. `python MLLM-5.2-Abyss-0P.py`, `python MLLM-5.2-Muir-26P.py`, `python MLLM-5.2-Monterey-71P.py`, `python MLLM-5.2-Tahoe-309P.py`, `python MLLM-5.2-Whitney-1042P.py` or `python MLLM-5.2-Golden-3981P.py` is the whole model — same engine as 5.1, just input and output.

Document Autocomplete that continues your document left-to-right. Type a prefix → dim ghost text inline → `Tab` to accept, `Esc` to dismiss. **Input → Ghost → Output.** No server, no install.

**[Try the live demo → mllm-editor.netlify.app](https://mllm-editor.netlify.app)** — all six models in a minimal Word-like document editor: ghost text as you type, `Tab` to accept, full decoding settings, everything local in your browser.

## Models

### Officially Released

| Model | File | Corpus | Status |
|---|---|---|---|
| **Abyss — 0P** | `MLLM-5.2-Abyss-0P.py` | BYO (`BUILT_IN_CORPUS` = `Placeholder`) | ✅ Released — Stable |
| **Muir — 26P** | `MLLM-5.2-Muir-26P.py` | ~26P embedded | ✅ Released — Stable |
| **Monterey — 71P** | `MLLM-5.2-Monterey-71P.py` | ~71P embedded | ✅ Released — Stable |
| **Tahoe — 309P** | `MLLM-5.2-Tahoe-309P.py` | ~309P embedded | ✅ Released — Stable |
| **Whitney — 1042P** | `MLLM-5.2-Whitney-1042P.py` | ~1042P embedded | ✅ Released — Stable |
| **Golden — 3981P** | `MLLM-5.2-Golden-3981P.py` | ~3981P embedded | ✅ Released — Stable — **NEW** — largest |

All six share same engine (`CausalTopology` + `AutocompleteEngine`), same flags, deterministic with `--seed`.

> Previously `MLLM-5.2-Muir-20P.py` → now `MLLM-5.2-Muir-26P.py` (same engine, re-counted).

Every model in the 5.2 lineup is now released — no upcoming models remain. Same engine, larger corpora — lightweight, local-first, zero-dep.

## What's new in 5.2 vs 5.1

Architecture unchanged from 5.1 — same topology and engine (`CausalTopology` + `AutocompleteEngine`). Corpus is autocomplete-based — ghost-text pairs (prefix → continuation) tuned for Tab-to-accept quality. Lightweight, pure Python 3.10+ stdlib.

**New in this release:** the full lineup is out — `Whitney — 1042P` (`MLLM-5.2-Whitney-1042P.py`, ~1042P embedded, ~2.8MB) and `Golden — 3981P` (`MLLM-5.2-Golden-3981P.py`, ~3981P embedded, ~11MB) — our largest corpus yet, same lightweight engine, ready to run. No server, no install — just `python MLLM-5.2-Golden-3981P.py "hello world"`.

## Quick start

Requires Python 3.10+, no `pip`.

```bash
# one-shot autocomplete (any model)
python MLLM-5.2-Abyss-0P.py "hello world" --steps 12 --seed 42
python MLLM-5.2-Muir-26P.py autocomplete "hello world" --steps 16 --seed 7
python MLLM-5.2-Monterey-71P.py autocomplete "hello world" --steps 16 --seed 7
python MLLM-5.2-Tahoe-309P.py autocomplete "hello world" --steps 16 --seed 7
python MLLM-5.2-Whitney-1042P.py autocomplete "hello world" --steps 16 --seed 7
python MLLM-5.2-Golden-3981P.py autocomplete "hello world" --steps 16 --seed 7

# interactive REPL (any file)
python MLLM-5.2-Tahoe-309P.py
# :help, :clear, :steps N, :temp N, :seed N, [quit] to exit

# pipe / stdin
echo "hello world" | python MLLM-5.2-Abyss-0P.py --steps 10 --plain
```

`--plain` prints `prefix + continuation` for pipes. Omit for ANSI ghost view with confidence heatmap.

> Previously `MLLM-5.2-Preview-0P.py` → now `MLLM-5.2-Abyss-0P.py` (same engine, stable name).

## Corpus — place to define

`MLLM-5.2-Abyss-0P.py` is BYO — edit the `BUILT_IN_CORPUS` placeholder at the top of the file and paste your autocomplete-based examples (prefix → continuation). `MLLM-5.2-Muir-26P.py` ships with ~26P embedded (~91KB), `MLLM-5.2-Monterey-71P.py` ships with ~71P embedded (~196KB), `MLLM-5.2-Tahoe-309P.py` ships with ~309P embedded (~830KB), `MLLM-5.2-Whitney-1042P.py` ships with ~1042P embedded (~2.8MB) and `MLLM-5.2-Golden-3981P.py` ships with ~3981P embedded (~11MB) and works out-of-the-box; same override applies.

- Format: UTF-8 text, one example per line ideal; paragraphs also work.
- Tokenization: `\b[a-zA-Z0-9']+\b|[.!?]` lowercased, split on `(?<=[.!?])\s+`.
- Sweet spot: 5KB–500KB of examples for immediate ghost quality.
- Optional override: `--corpus path/to/file.txt` for a one-off external file (not primary).

## Flags

| Flag | Default | Description |
|---|---|---|
| `--steps N` | `16` | Amt of iterations |
| `--temperature T` | `0.35` | 0.0 precise → 1.2 creative |
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
MLLM-5.2-Abyss-0P.py     ← 0P, BYO placeholder — self-contained runner, zero-dep, BUILT_IN_CORPUS = Placeholder
MLLM-5.2-Muir-26P.py     ← 26P, ~91KB embedded — same engine, ready to run
MLLM-5.2-Monterey-71P.py ← 71P, ~196KB embedded — same engine, ready to run
MLLM-5.2-Tahoe-309P.py   ← 309P, ~830KB embedded — same engine, ready to run
MLLM-5.2-Whitney-1042P.py ← 1042P, ~2.8MB embedded — same engine, ready to run
MLLM-5.2-Golden-3981P.py  ← 3981P, ~11MB embedded — same engine, largest yet, ready to run
README.md                  ← this file
editor/                    ← browser demo — Word-like ghost-text editor, all six models (live at mllm-editor.netlify.app)
LICENSE                    ← MIT
```

## Roadmap

**Released (Stable)**

- `Abyss — 0P` (`MLLM-5.2-Abyss-0P.py`) — BYO placeholder — ✅ Released
- `Muir — 26P` (`MLLM-5.2-Muir-26P.py`) — ~26P embedded (~91KB) — ✅ Released
- `Monterey — 71P` (`MLLM-5.2-Monterey-71P.py`) — ~71P embedded (~196KB) — ✅ Released
- `Tahoe — 309P` (`MLLM-5.2-Tahoe-309P.py`) — ~309P embedded (~830KB) — ✅ Released
- `Whitney — 1042P` (`MLLM-5.2-Whitney-1042P.py`) — ~1042P embedded (~2.8MB) — ✅ Released
- `Golden — 3981P` (`MLLM-5.2-Golden-3981P.py`) — ~3981P embedded (~11MB) — ✅ Released — **NEW** — largest

**All models have been released.** Same engine (`CausalTopology` + `AutocompleteEngine`) across the whole lineup — lightweight, local-first, zero-dep.

**Document Editor — live at [mllm-editor.netlify.app](https://mllm-editor.netlify.app).** The MLLM document editor is a browser editor built around 5.2 ghost-text: inline ghost completions as you type, `Tab` / `Esc` to accept or dismiss, per-token confidence shown as ghost opacity, and steps / temperature / threshold / max-ngram / seed controls — all six models run locally in a Web Worker (causal n-gram engine ported from the Python originals), no server. Source in `editor/` (`python3 editor/extract_corpora.py` regenerates the corpus files it serves).

---

*Stable release: causal n-gram, autocomplete ghost-text, confidence-gated, deterministic with --seed, lightweight. All six models — Abyss 0P through Golden 3981P — now available. MLLM document editor live at [mllm-editor.netlify.app](https://mllm-editor.netlify.app).*
