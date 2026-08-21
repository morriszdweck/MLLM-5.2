# MLLM-5.2

> 🚧 **Preview — MLLM-5.2 Preview Release**
> Preview artifact — APIs and behavior may change before stable. Corpus included — ready to run.

**Lightweight Autocomplete Micro Language Models — preview included, ready to run.**

**Document Autocomplete Micro Language Model** — tiny causal n-gram that **continues your document left-to-right**. Type a prefix → dim ghost-text inline → `Tab` to accept, `Esc` to dismiss — like Copilot for prose. **Input → Ghost → Output.** No server, no install — just `python MLLM-5.2.py` with lightweight autocomplete models included.

> **Single-file, zero-dependency.** `python MLLM-5.2.py` is the whole model — same engine as 5.1, just input and output. Corpus placeholder included, deterministic, local-first.

## What's new in 5.2 vs 5.1

**Architecture unchanged from 5.1** — causal + bidirectional **topology/engine identical to 5.1** (`CausalTopology` + `AutocompleteEngine`, same scoring/softmax/sampling). No engine rewrite — 5.2 is the same lightweight autocomplete preview on the same topology.

- **Same topology/engine** — `left_counts[n][ctx][word]` causal n-gram, `log(count/total)*n` scoring, `softmax → p^(1/T)` sampling identical to 5.1.
- **Corpus is autocomplete-based** — ships with **autocomplete-based ghost-text examples** (prefix → continuation pairs for Tab-to-accept), not encyclopedia. Tuned for inline ghost quality.
- **Lightweight preview** — single-file, zero-dep, instant startup. Lightweight autocomplete models included.

> Positioning: **lightweight autocomplete models** — not LLMs. Tiny, local, deterministic. Same engine as 5.1; only the corpus is now autocomplete-based.

## Quick start — Python

> **Preview ready:** `python MLLM-5.2.py --corpus "MLLM 5.2 preview" "hello world"`

Requires **Python 3.10+**, no `pip` — pure stdlib. Ghost UX: cyan prefix + dim ghost → `Tab`/`Esc`, confidence heatmap below.

```bash
# one-shot autocomplete (corpus included — preview ready)
python MLLM-5.2.py --corpus "MLLM 5.2 preview" "hello world" --steps 12 --seed 42
python MLLM-5.2.py --corpus "MLLM 5.2 preview" autocomplete "hello world" --steps 16 --seed 7

# interactive REPL
python MLLM-5.2.py --corpus "MLLM 5.2 preview"
# inside: :help, :clear, :steps N, :temp N, :seed N, [quit] to exit

# pipe / stdin
echo "hello world" | python MLLM-5.2.py --corpus "MLLM 5.2 preview" --steps 10 --plain
```

`--plain` prints `prefix + continuation` for pipes. Omit for ANSI ghost view.

## Corpus — place to define

No custom corpus pipeline — `MLLM-5.2.py` is simply a **place to define your corpus**. Edit the `BUILT_IN_CORPUS` placeholder in `MLLM-5.2.py` (mirrored in `MLLM 5.2 preview` file) and paste your **autocomplete-based ghost-text examples** there — prefix → continuation pairs, not encyclopedia entries.

- **Format:** plain `.txt`, UTF-8. One example per line ideal (`prefix → ghost` style, `(?<=[.!?])\s+` split); paragraphs work.
- **Tokenization:** `\b[a-zA-Z0-9']+\b|[.!?]` lowercased.
- **Sweet spot:** **5KB–500KB** of ghost-text examples — tuned for immediate Tab-to-accept quality.
- **Optional override:** `--corpus PATH` still works for a one-off external file (`python MLLM-5.2.py --corpus path/to/file.txt "hello"`), but primary flow is editing the placeholder.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--corpus PATH` | `"MLLM 5.2 preview"` | Optional override — external `.txt`; default uses built-in placeholder |
| `--steps N` | `16` | Max tokens to generate |
| `--temperature T` | `0.35` | `0.0` greedy → `1.2` creative |
| `--threshold T` | `0.0` | Min confidence to emit |
| `--max-ngram N` | `3` | Causal context size (1–5) |
| `--seed N` | random | Deterministic sampling |
| `--no-color` | auto | Disable ANSI (or `NO_COLOR=1`) |
| `--plain` | off | Plain output for pipes |

Subcommands `autocomplete`/`generate`/`complete`/`chat` share flags. Bare prefix also works: `python MLLM-5.2.py --corpus "MLLM 5.2 preview" "hello"` == `... autocomplete "hello"`.

## How it works

1. **Topology.** Tokenize lowercased `\b[a-zA-Z0-9']+\b|[.!?]`; build causal `left_counts[n][ctx][word]` for `n=1..3` (`CausalTopology`).
2. **Score.** `log(count/total + 1e-5)*n` summed over matching `n` + `log(unigram+1)*0.1` backoff.
3. **Softmax & sample.** `softmax(score) → p(w)`, sampled via `p^(1/T)` with `random.Random(seed)`; `T≤0` greedy. Deterministic across `PYTHONHASHSEED`.
4. **Threshold.** Stop if `p < threshold`; else append and continue to `--steps` or `.!?` after ≥4 tokens.

Architecture identical to 5.1 — `BidirectionalTopology` + `DiscreteDiffusionEngine`/`AutocompleteEngine.complete()`.

## Project layout

```
MLLM-5.2.py           ← self-contained runner — same architecture as 5.1, zero-dep, input → output
MLLM 5.2 preview      ← corpus placeholder mirror (UTF-8 .txt, autocomplete-based)
README.md             ← this file
```

## Roadmap / Coming Soon

> ✨ **MLLM Document Editor — coming soon** ✨
> Full editor built around 5.2 ghost-text — type → dim ghost → `Tab` to accept, `Esc` to dismiss.

Lightweight autocomplete was just the start. The **MLLM Document Editor** is document-first with 5.2's ghost at its core: inline ghost, `Tab`/`Ctrl+→`/`Esc`, confidence heatmap, temperature/length controls — all local, instant, powered by the included preview corpus.

> **Coming soon** — single `index.html` + `MLLM-5.2.py` engine. Preview polish underway. Other lightweight autocomplete models remain tiny and local — 5.2 Preview is the base ghost; the editor is its home.

---

*Core principles: causal n-gram topology, autocomplete ghost-text, confidence-gated, deterministic with `--seed`, lightweight autocomplete models included — preview ready, input → output.*
