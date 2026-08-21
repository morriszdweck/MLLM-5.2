# MLLM-5.2

**Lightweight Autocomplete Micro Language Models — your corpus, your ghost.**

**Document Autocomplete Micro Language Model** — tiny causal n-gram that **continues your document left-to-right**. Type a prefix → dim ghost-text inline → `Tab` to accept, `Esc` to dismiss — like Copilot for prose. **Input → Ghost → Output.** No server, no install — just `python MLLM-5.2.py` + your corpus.

> **Single-file, zero-dependency.** `python MLLM-5.2.py` is the whole model. Your corpus is the personality. Lightweight autocomplete, not a chatbot.

```
 ███╗   ███╗██╗     ██╗     ███╗   ███╗      ███████╗   ██╗  ██╗
 ████╗ ████║██║     ██║     ████╗ ████║      ██╔════╝   ██║  ██║
 ██╔████╔██║██║     ██║     ██╔████╔██║█████╗███████╗   ███████║
 ██║╚██╔╝██║██║     ██║     ██║╚██╔╝██║╚════╝╚════██║   ╚════██║
 ██║ ╚═╝ ██║███████╗███████╗██║ ╚═╝ ██║      ███████║██╗██║  ██║
 ╚═╝     ╚═╝╚══════╝╚══════╝╚═╝     ╚═╝      ╚══════╝╚═╝╚═╝  ╚═╝
       Lightweight · BYO Corpus · Input → Ghost → Output
```

## What's new in 5.2 vs 5.1

MLLM-5.2 is a major simplification — **lighter, faster, and BYO corpus**:

- **Removed embedded corpus** — no `BUILT_IN_CORPUS`. 5.1 shipped with `data/corpus.txt` baked in; 5.2 ships with *no* corpus. You provide it via `--corpus`. Smaller repo, no stale data, your style from day one.
- **Simplified to Input → Output** — no diffusion, no `denoise()`/`--show-steps`, no right-context. Just `input → ghost → output`: causal left context does one job well. Single autocomplete path (aliases `autocomplete`/`generate`/`complete` still work).
- **Lighter & faster** — single-file, zero-dep, **<500 lines** (vs 1817 in 5.1), instant startup, no build. **BYO corpus via `--corpus`** — point at any `.txt` (try `MLLM 5.2 preview` included as template) and go.

> Positioning: **lightweight autocomplete models** — not LLMs. Tiny, local, deterministic when you want it, yours to shape. If 5.1 was the playground, 5.2 is the engine.

## Quick start — Python

> **BYO corpus in 2 steps:** ① Edit `MLLM 5.2 preview` (delete EXAMPLE, paste your text) → ② Run with `--corpus "MLLM 5.2 preview"`.

Requires **Python 3.10+**, no `pip`, no `venv` — pure stdlib. **Ghost UX:** cyan prefix + dim ghost-text → `Tab` accepts, `Esc` dismisses, confidence heatmap below.

```bash
# one-shot autocomplete (BYO corpus — use the included preview file)
python MLLM-5.2.py --corpus "MLLM 5.2 preview" "hello world" --steps 12 --seed 42
python MLLM-5.2.py --corpus "MLLM 5.2 preview" "the quick brown" --temperature 0.3 --threshold 0.15

# aliases still work
python MLLM-5.2.py --corpus "MLLM 5.2 preview" autocomplete "hello world" --steps 16 --seed 7
python MLLM-5.2.py --corpus your.txt generate "once upon a time" --steps 16

# interactive REPL — type prefix → ghost continuation (Tab to accept in editor, Enter to re-prompt here)
python MLLM-5.2.py --corpus "MLLM 5.2 preview"
python MLLM-5.2.py --corpus your.txt
# inside REPL: :help, :clear, :steps N, :temp N, :seed N, [quit] to exit

# pipe / stdin (great for shell chains)
echo "hello world" | python MLLM-5.2.py --corpus "MLLM 5.2 preview" --steps 10 --plain
cat notes.txt | python MLLM-5.2.py --corpus "MLLM 5.2 preview" --steps 20 --temperature 0.35 --plain
```

`--plain` prints just `prefix + continuation` for pipes/tests. Omit it for the ANSI ghost view (cyan prefix + dim ghost, confidence heatmap).

On Homebrew Python, no venv needed — it's stdlib only. If you use a venv anyway, `python MLLM-5.2.py` still works.

## Bring your own corpus

**2 steps:** **1)** Edit `MLLM 5.2 preview` — delete EXAMPLE block, paste your text below `### PASTE YOUR CORPUS BELOW THIS LINE ###` (or copy to `my-corpus.txt`). **2)** Run with `--corpus`.

5.2 has **no embedded corpus** — you bring it.

- **Format:** plain `.txt`, UTF-8. One sentence per line is ideal (sentence-split on `(?<=[.!?])\s+`), but paragraphs work — whitespace is normalized.
- **Tokenization:** `\b[a-zA-Z0-9']+\b|[.!?]` lowercased. So `Hello!` → `hello`, `!`.
- **Sweet spot:** **5KB–500KB** (a few hundred to a few thousand sentences). Too small = repetitive ghosts; too large = diluted n-grams but still fast.
- **Template:** included **`MLLM 5.2 preview`** is the template — keep the filename (spaces, no extension) or duplicate to `my-corpus.txt`, then point `--corpus` at it:

```bash
cp "MLLM 5.2 preview" my-corpus.txt
# edit my-corpus.txt — one sentence per line works best
python MLLM-5.2.py --corpus my-corpus.txt "my opening line" --steps 16 --seed 42
```

No corpus found error:

```
mllm52: error: No corpus found — provide --corpus your.txt or place text in 'MLLM 5.2 preview' and run --corpus 'MLLM 5.2 preview'
```

> Tip: keep corpora focused. A 20KB corpus in your voice beats a 2MB mixed dump for ghost quality.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--corpus PATH` | *(required)* | Path to training corpus `.txt` — e.g. `"MLLM 5.2 preview"` or `your.txt` |
| `--steps N` | `16` | Max tokens to generate (effort knob) |
| `--temperature T` | `0.35` | `0.0` greedy → `1.2` creative |
| `--threshold T` | `0.0` | Min confidence to emit (gates low-confidence ghosts) |
| `--max-ngram N` | `3` | Causal context size (1–5) |
| `--seed N` | random | Deterministic sampling (sorted candidates → reproducible) |
| `--no-color` | auto | Disable ANSI (or `NO_COLOR=1`) |
| `--plain` | off | Plain output: just print continuation (for pipes) |

Subcommands `autocomplete`, `generate`, `complete`, `chat` all accept the same flags (subcommand flags override top-level). Bare prefix mode also works: `python MLLM-5.2.py --corpus "MLLM 5.2 preview" "hello"` equals `... autocomplete "hello"`.

## How it works

1. **Topology.** Tokenize `\b[a-zA-Z0-9']+\b|[.!?]` lowercased, sentence-split `(?<=[.!?])\s+`. Build **causal** left counts `left_counts[n][ctx][word]` for `n=1..3` (configurable via `--max-ngram`). Stored in `CausalTopology` (`MLLM-5.2.py`).

2. **Score.** For each position given left tokens `w_{<t}`, score each candidate `w` by `log(count/total + FLOOR)*n` summed over `n=1..3` where context matches, plus `log(unigram+1)*0.1` backoff. `FLOOR=1e-5`.

3. **Softmax & sample.** `softmax(score) → p(w)`. Sample with `p^(1/T)` using `random.Random(seed)` where `T=temperature`; `T≤0` is greedy (`argmax`). Candidates = `sorted(observed ∪ top50 unigrams)` so deterministic across `PYTHONHASHSEED`.

4. **Threshold.** If chosen token's `p < threshold`, stop emitting — ghost is gated. Otherwise append and continue left-to-right up to `--steps` or sentence end (`.!?` after ≥4 tokens).

Single file, ~481 lines, zero dependencies — the whole engine is `CausalTopology` + `AutocompleteEngine.complete()`.

## Project layout

```
MLLM-5.2.py           ← self-contained autocomplete runner (run it) — 481 lines, zero-dep
MLLM 5.2 preview      ← BYO corpus template (plain UTF-8 .txt, one sentence per line) — copy & edit
mllm52/               ← (optional) installable package mirror of MLLM-5.2.py if present
  topology.py           n-gram topology (if split)
  engine.py             AutocompleteEngine
netlify.toml          ← static deploy config if site is published (no build)
tests/                ← pytest suite if present
README.md             ← this file
```

No `data/corpus.txt` — 5.2 has no embedded corpus. You create it. Point `--corpus` at `"MLLM 5.2 preview"` or your own file.

## Roadmap / Coming Soon

> ✨ **MLLM Document Editor — coming soon** ✨
> A **full editor built around 5.2 ghost-text** — type → dim ghost → `Tab` to accept, `Esc` to dismiss.

The lightweight autocomplete model was just the start. The **MLLM Document Editor** is a document-first editor with 5.2's ghost at its core: type, see the dim continuation inline (like Copilot for prose), `Tab` to accept, `Ctrl+→` for word, `Esc` to dismiss — all local, instant, and trained on *your* corpus (`MLLM 5.2 preview` or your own). Minimal surface + causal ghost, confidence heatmap, temperature/length controls, one-click rebuild.

> **Coming soon** — single `index.html` (like 5.1) + `MLLM-5.2.py` engine. Your corpus, your ghost, your document.

Other lightweight autocomplete models in the family remain tiny and local — 5.2 is the base ghost; the editor is the home for it. No cloud, no keys, just text in → ghost out.

## Deploy (optional)

5.2 is Python-focused, but if you ship a playground `index.html`, it's static — no build:

```toml
# netlify.toml
[build]
  publish = "."
  command = "echo 'no build - static deploy'"
```

```bash
npx netlify deploy --prod --dir=. --site <your-site>
open index.html  # local, offline, file:// works
```

---

*Core principles: causal n-gram topology, autocomplete ghost-text, confidence-gated, deterministic with `--seed`, and now BYO corpus — your corpus, your ghost.*
