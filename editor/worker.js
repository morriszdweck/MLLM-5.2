/* MLLM-5.2 editor worker — builds topologies and answers completion requests
 * off the main thread so large corpora (Golden ~11MB) don't freeze the UI. */
importScripts('engine.js');

const M = self.MLLM;
let topo = null;
let engine = null;
let builtFor = null; // { modelId, maxN }

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.cmd === 'build') {
    try {
      let text = msg.corpusText;
      if (text == null) {
        postMessage({ type: 'phase', modelId: msg.modelId, phase: 'fetch' });
        const res = await fetch(msg.url);
        if (!res.ok) throw new Error('could not fetch corpus (HTTP ' + res.status + ')');
        text = await res.text();
      }
      postMessage({ type: 'phase', modelId: msg.modelId, phase: 'build' });
      const t = new M.Topology(msg.maxN);
      t.ingest(text, {
        onBatch: (sentences) =>
          postMessage({ type: 'progress', modelId: msg.modelId, sentences }),
      });
      topo = t;
      engine = new M.AutocompleteEngine(topo);
      builtFor = { modelId: msg.modelId, maxN: msg.maxN };
      postMessage({
        type: 'built',
        modelId: msg.modelId,
        stats: {
          vocab: topo.vocab.size,
          tokens: topo.tokens,
          sentences: topo.sentences,
          maxN: msg.maxN,
        },
      });
    } catch (err) {
      postMessage({
        type: 'error',
        modelId: msg.modelId,
        message: String((err && err.message) || err),
      });
    }
    return;
  }

  if (msg.cmd === 'complete') {
    const ready =
      engine && builtFor && builtFor.modelId === msg.modelId && builtFor.maxN >= (msg.opts.maxN || builtFor.maxN);
    if (!ready) {
      postMessage({ type: 'result', id: msg.id, continuation: [], confidences: [] });
      return;
    }
    engine.rng = msg.opts.seed == null ? Math.random : M.mulberry32(msg.opts.seed);
    const res = engine.complete(msg.prefix, msg.opts);
    postMessage({
      type: 'result',
      id: msg.id,
      continuation: res.continuation,
      confidences: res.confidences,
    });
  }
};
