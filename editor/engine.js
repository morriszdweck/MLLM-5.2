/* MLLM-5.2 engine — faithful JavaScript port of the causal n-gram autocomplete
 * engine shared by all six models (BidirectionalTopology + DiscreteDiffusionEngine,
 * autocomplete path only). Mirrors the Python semantics exactly, including the
 * tokenizer's Unicode-aware word boundaries, the log-score accumulation, and the
 * softmax/candidate ordering, so greedy output (temperature = 0) matches the CLI.
 *
 * One deliberate difference: seeded sampling uses a local PRNG (mulberry32), so
 * --seed is reproducible within the web app but produces a different stream than
 * Python's random.Random.
 */
(function (global) {
  'use strict';

  const FLOOR = 1e-5;
  const UNIGRAM_WEIGHT = 0.1;
  const BACKOFF_SIZE = 50;
  const SEP = '\u0001';
  const SENT_END = new Set(['.', '!', '?']);

  // Python: re.compile(r"\b[a-zA-Z0-9']+\b|[.!?]") with Unicode-aware \b.
  // JS \b is ASCII-only, so candidates are filtered with explicit Unicode
  // boundary checks to reproduce CPython findall behavior exactly.
  const CANDIDATE_RE = /[a-zA-Z0-9']+|[.!?]/g;
  const WORD_CHAR_RE = /[\p{L}\p{N}_]/u;

  function isWordChar(ch) {
    return ch !== undefined && WORD_CHAR_RE.test(ch);
  }

  function tokenize(text) {
    const s = text.toLowerCase();
    const out = [];
    CANDIDATE_RE.lastIndex = 0;
    let m;
    while ((m = CANDIDATE_RE.exec(s)) !== null) {
      const tok = m[0];
      if (tok.length === 1 && SENT_END.has(tok)) {
        out.push(tok);
        continue;
      }
      const start = m.index;
      const end = start + tok.length;
      const before = start > 0 ? s[start - 1] : undefined;
      const after = end < s.length ? s[end] : undefined;
      // \b requires: not preceded/followed by a Unicode word char
      if (isWordChar(before) || isWordChar(after)) continue;
      out.push(tok);
    }
    return out;
  }

  function splitSentences(flat) {
    return flat.split(/(?<=[.!?])\s+/);
  }

  // mulberry32 — small deterministic PRNG for seeded sampling
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  class Topology {
    constructor(maxN) {
      if (maxN < 1) throw new Error('max_n must be >= 1');
      this.maxN = maxN;
      this.left = []; // left[n]: Map<ctxKey, Map<word, count>>
      this.leftTotals = []; // leftTotals[n]: Map<ctxKey, number>
      for (let n = 0; n <= maxN; n++) {
        this.left.push(new Map());
        this.leftTotals.push(new Map());
      }
      this.unigrams = new Map();
      this.vocab = new Set();
      this.sentences = 0;
      this.tokens = 0;
    }

    static fromText(text, maxN, opts) {
      const topo = new Topology(maxN);
      topo.ingest(text, opts);
      return topo;
    }

    ingest(text, opts) {
      if (!text || !text.trim()) throw new Error('corpus text is empty');
      const onBatch = opts && opts.onBatch;
      const flat = text.replace(/\s+/g, ' ').trim();
      let ingested = 0;
      let sinceBatch = 0;
      for (const sentence of splitSentences(flat)) {
        const words = tokenize(sentence);
        if (words.length === 0) continue;
        this.sentences++;
        sinceBatch++;
        if (onBatch && sinceBatch >= 512) {
          onBatch(this.sentences);
          sinceBatch = 0;
        }
        ingested += words.length;
        for (const w of words) {
          this.vocab.add(w);
          this.unigrams.set(w, (this.unigrams.get(w) || 0) + 1);
        }
        for (let n = 1; n <= this.maxN; n++) {
          const counts = this.left[n];
          const totals = this.leftTotals[n];
          for (let i = n; i < words.length; i++) {
            let ctx;
            if (n === 1) ctx = words[i - 1];
            else ctx = words.slice(i - n, i).join(SEP);
            let bucket = counts.get(ctx);
            if (bucket === undefined) {
              bucket = new Map();
              counts.set(ctx, bucket);
            }
            const target = words[i];
            bucket.set(target, (bucket.get(target) || 0) + 1);
            totals.set(ctx, (totals.get(ctx) || 0) + 1);
          }
        }
      }
      this.tokens += ingested;
      if (ingested === 0) throw new Error('corpus text contains no usable tokens');
    }
  }

  class AutocompleteEngine {
    constructor(topo, rng) {
      this.topo = topo;
      this.rng = rng || Math.random;
      // Python: [w for w, _ in topo.unigrams.most_common(50)]
      // Counter.most_common ties break by insertion order; replicate with a
      // stable sort by count desc.
      this.backoff = Array.from(topo.unigrams.entries())
        .map(([word, count], idx) => ({ word, count, idx }))
        .sort((a, b) => b.count - a.count || a.idx - b.idx)
        .slice(0, BACKOFF_SIZE)
        .map((e) => e.word);
    }

    // Causal (left-only) softmax distribution for the position at the end of seq.
    causalDistribution(seq, maxN) {
      const idx = seq.length;
      const topo = this.topo;
      const limit = Math.min(maxN || topo.maxN, topo.maxN);
      let base = 0.0;
      const contrib = new Map();
      for (let n = 1; n <= limit; n++) {
        if (idx < n) continue;
        const ctx =
          n === 1 ? seq[idx - 1] : seq.slice(idx - n, idx).join(SEP);
        const counts = topo.left[n].get(ctx);
        const total = topo.leftTotals[n].get(ctx) || 0;
        if (!counts || total <= 0) continue;
        const floorAdj = Math.log(FLOOR) * n;
        base += floorAdj;
        for (const [word, count] of counts) {
          contrib.set(
            word,
            (contrib.get(word) || 0) + (Math.log(count / total + FLOOR) * n - floorAdj)
          );
        }
      }
      // candidates = sorted(set(contrib) | set(backoff)) — sorted by code point
      const candSet = new Set(contrib.keys());
      for (const w of this.backoff) candSet.add(w);
      const candidates = Array.from(candSet).sort();
      if (candidates.length === 0) return null;
      const energies = new Map();
      let maxE = -Infinity;
      for (const w of candidates) {
        const uni = topo.unigrams.has(w) ? topo.unigrams.get(w) : 1;
        const e =
          Math.log(uni + 1) * UNIGRAM_WEIGHT + base + (contrib.get(w) || 0);
        energies.set(w, e);
        if (e > maxE) maxE = e;
      }
      let total = 0;
      const probs = new Map();
      for (const w of candidates) {
        const x = Math.exp(energies.get(w) - maxE);
        probs.set(w, x);
        total += x;
      }
      for (const [w, x] of probs) probs.set(w, x / total);
      return probs; // Map in sorted-candidate order, like Python's dict
    }

    // Left-to-right autocomplete: continue prefix token by token.
    complete(prefix, opts) {
      const maxTokens = opts && opts.maxTokens != null ? opts.maxTokens : 12;
      const temperature = opts && opts.temperature != null ? opts.temperature : 0.35;
      const threshold = opts && opts.threshold != null ? opts.threshold : 0.0;
      const maxN = opts && opts.maxN != null ? opts.maxN : undefined;
      const seq = prefix.map((w) => w.toLowerCase());
      const continuation = [];
      const confidences = [];
      for (let step = 0; step < maxTokens; step++) {
        const probs = this.causalDistribution(seq, maxN);
        if (!probs) break;
        let chosen, conf;
        if (temperature <= 0) {
          chosen = null;
          conf = -1;
          for (const [w, p] of probs) {
            if (p > conf) {
              conf = p;
              chosen = w;
            }
          }
        } else {
          const t = Math.max(temperature, 1e-6);
          let totalW = 0;
          const entries = [];
          for (const [w, p] of probs) {
            const weight = Math.pow(p, 1 / t);
            totalW += weight;
            entries.push([w, weight, p]);
          }
          let r = this.rng() * totalW;
          chosen = entries[entries.length - 1][0];
          conf = entries[entries.length - 1][2];
          for (const [w, weight, p] of entries) {
            r -= weight;
            if (r <= 0) {
              chosen = w;
              conf = p;
              break;
            }
          }
        }
        if (conf < threshold) break;
        seq.push(chosen);
        continuation.push(chosen);
        confidences.push(conf);
        if (SENT_END.has(chosen) && continuation.length >= 4) break;
      }
      return { continuation, confidences };
    }
  }

  // Mirrors assemble_autocomplete's continuation handling: joins tokens,
  // tightens punctuation spacing, capitalizes after sentence boundaries.
  function assembleContinuation(prefixWords, contWords) {
    let cont = contWords.join(' ');
    for (const mark of [',', '.', '!', '?']) {
      cont = cont.split(' ' + mark).join(mark);
    }
    const prefixStr = prefixWords.join(' ');
    const prefixEndsSentence =
      prefixStr.length === 0 || /[.!?]\s*$/.test(prefixStr);
    if (cont && prefixEndsSentence) {
      cont = cont[0].toUpperCase() + cont.slice(1);
    }
    return cont;
  }

  const MLLM = {
    tokenize,
    splitSentences,
    Topology,
    AutocompleteEngine,
    assembleContinuation,
    mulberry32,
    FLOOR,
    UNIGRAM_WEIGHT,
    BACKOFF_SIZE,
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = MLLM;
  } else {
    global.MLLM = MLLM;
  }
})(typeof window !== 'undefined' ? window : globalThis);
