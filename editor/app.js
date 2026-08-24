/* MLLM Editor — Word-like document editor wired to the MLLM-5.2 engine.
 * All models run locally in a Web Worker; ghost text renders inline at the caret.
 */
'use strict';

/* ── model registry ─────────────────────────────────────────── */
const MODELS = [
  { id: 'abyss', name: 'Abyss — 0P', size: 'BYO', url: null,
    meta: 'Bring your own corpus' },
  { id: 'muir', name: 'Muir — 26P', size: '~91KB', url: 'corpora/muir.txt',
    meta: 'Smallest embedded corpus · instant' },
  { id: 'monterey', name: 'Monterey — 71P', size: '~196KB', url: 'corpora/monterey.txt',
    meta: 'Small corpus · fast' },
  { id: 'tahoe', name: 'Tahoe — 309P', size: '~830KB', url: 'corpora/tahoe.txt',
    meta: 'Mid-size corpus · good default' },
  { id: 'whitney', name: 'Whitney — 1042P', size: '~2.8MB', url: 'corpora/whitney.txt',
    meta: 'Large corpus · a few seconds first load' },
  { id: 'golden', name: 'Golden — 3981P', size: '~11MB', url: 'corpora/golden.txt',
    meta: 'Largest corpus · slowest first load' },
];
const modelById = (id) => MODELS.find((m) => m.id === id);

const ABYSS_SAMPLE = `The quick brown fox jumps over the lazy dog.
A document editor helps you write, edit, and format text.
Good writing is clear writing, and clear writing takes rewriting.
The best way to start a document is to start typing.
Every sentence you write can be improved by the sentence that follows it.
Autocomplete suggests the next words based on the words before them.
Tab accepts the suggestion, and Escape sends it away.
A small model learns from a small corpus, so choose your examples well.
The editor saves your document locally as you work.
Writing tools should stay out of the way until you need them.`;

/* ── settings ───────────────────────────────────────────────── */
const DEFAULT_SETTINGS = {
  modelId: 'tahoe',
  steps: 1,
  temperature: 0.35,
  threshold: 0,
  maxNgram: 3,
  seed: null, // null = random
  autocomplete: true,
};
// bump when defaults change in a way existing visitors should pick up
const SETTINGS_VERSION = 2;
let settings = { ...DEFAULT_SETTINGS };
try {
  const saved = JSON.parse(localStorage.getItem('mllm.settings') || '{}');
  if (saved.version === SETTINGS_VERSION) settings = { ...DEFAULT_SETTINGS, ...saved };
} catch (_) { /* fresh start */ }
if (!modelById(settings.modelId)) settings.modelId = 'tahoe';

function saveSettings() {
  localStorage.setItem('mllm.settings', JSON.stringify({ ...settings, version: SETTINGS_VERSION }));
}

/* ── dom refs ───────────────────────────────────────────────── */
const page = document.getElementById('page');
const docTitle = document.getElementById('docTitle');
const ribbon = document.getElementById('ribbon');
const modelSelect = document.getElementById('modelSelect');
const modelCards = document.getElementById('modelCards');
const settingsPanel = document.getElementById('settings');
const wordCountEl = document.getElementById('wordCount');
const modelStatusEl = document.getElementById('modelStatus');
const ghostStatusEl = document.getElementById('ghostStatus');
const abyssBox = document.getElementById('abyssBox');
const abyssCorpus = document.getElementById('abyssCorpus');

/* ── document persistence ───────────────────────────────────── */
let saveTimer = null;
function saveSoon() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    localStorage.setItem('mllm.doc', page.innerHTML);
    localStorage.setItem('mllm.title', docTitle.value);
  }, 700);
}
(function restore() {
  const html = localStorage.getItem('mllm.doc');
  const title = localStorage.getItem('mllm.title');
  if (html) page.innerHTML = html;
  page.querySelectorAll('.ghost').forEach((g) => g.remove()); // never persist ghosts
  if (title) docTitle.value = title;
})();

/* ── ribbon: formatting ─────────────────────────────────────── */
const FMT_STATE_CMDS = [
  'bold', 'italic', 'underline', 'strikeThrough',
  'insertUnorderedList', 'insertOrderedList',
  'justifyLeft', 'justifyCenter', 'justifyRight', 'justifyFull',
];

function selectionInPage() {
  const sel = document.getSelection();
  if (!sel.rangeCount) return false;
  return page.contains(sel.anchorNode);
}

function exec(cmd, value = null) {
  document.execCommand(cmd, false, value);
  updateFmtStates();
  updateCounts();
  saveSoon();
}

function updateFmtStates() {
  if (!selectionInPage()) return;
  ribbon.querySelectorAll('button[data-cmd]').forEach((btn) => {
    const c = btn.dataset.cmd;
    if (!FMT_STATE_CMDS.includes(c)) return;
    let on = false;
    try { on = document.queryCommandState(c); } catch (_) { /* unsupported */ }
    btn.classList.toggle('active', !!on);
  });
  const block = document.getElementById('blockFormat');
  try {
    const v = String(document.queryCommandValue('formatBlock') || 'p').toLowerCase();
    block.value = ['p', 'h1', 'h2', 'h3', 'blockquote'].includes(v) ? v : 'p';
  } catch (_) { /* keep */ }
}

ribbon.addEventListener('mousedown', (e) => {
  // keep the editor selection while using ribbon buttons / color wells
  if (e.target.closest('button') || e.target.closest('.color-btn')) e.preventDefault();
});

ribbon.querySelectorAll('button[data-cmd]').forEach((btn) => {
  btn.addEventListener('click', () => exec(btn.dataset.cmd));
});

document.getElementById('blockFormat').addEventListener('change', (e) => {
  exec('formatBlock', e.target.value);
  page.focus();
});
document.getElementById('fontName').addEventListener('change', (e) => {
  if (e.target.value) exec('fontName', e.target.value);
  e.target.value = '';
  page.focus();
});
document.getElementById('fontSize').addEventListener('change', (e) => {
  const px = e.target.value;
  e.target.value = '';
  if (!px) return;
  // execCommand fontSize only takes 1–7; emit size 7 then restyle the <font> tags
  document.execCommand('fontSize', false, '7');
  page.querySelectorAll('font[size="7"]').forEach((f) => {
    const span = document.createElement('span');
    span.style.fontSize = px + 'px';
    span.innerHTML = f.innerHTML;
    f.replaceWith(span);
  });
  updateCounts();
  saveSoon();
  page.focus();
});
document.getElementById('foreColor').addEventListener('input', (e) => exec('foreColor', e.target.value));
document.getElementById('hiliteColor').addEventListener('input', (e) => exec('hiliteColor', e.target.value));

/* ── title bar actions ──────────────────────────────────────── */
function slug() {
  return (docTitle.value.trim() || 'document').replace(/[^\w\- ]+/g, '').trim().replace(/\s+/g, '-').toLowerCase() || 'document';
}
function download(filename, text, type) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
document.getElementById('btnDownloadTxt').addEventListener('click', () => {
  dismissGhost(); // don't leak the suggestion into the file
  download(slug() + '.txt', page.innerText, 'text/plain');
});
document.getElementById('btnDownloadHtml').addEventListener('click', () => {
  dismissGhost();
  download(slug() + '.html',
    '<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8">\n<title>' +
    docTitle.value.replace(/</g, '&lt;') + '</title>\n</head>\n<body>\n' +
    page.innerHTML + '\n</body>\n</html>', 'text/html');
});
document.getElementById('btnNew').addEventListener('click', () => {
  if (!confirm('Start a new document? The current one will be cleared.')) return;
  dismissGhost();
  page.innerHTML = '';
  docTitle.value = 'Untitled document';
  localStorage.removeItem('mllm.doc');
  localStorage.removeItem('mllm.title');
  updateCounts();
  page.focus();
});
docTitle.addEventListener('input', saveSoon);

/* ── status bar ─────────────────────────────────────────────── */
function updateCounts() {
  // count text nodes, skipping any live ghost suggestion
  const walker = document.createTreeWalker(page, NodeFilter.SHOW_TEXT, {
    acceptNode: (n) =>
      n.parentElement && n.parentElement.closest('.ghost')
        ? NodeFilter.FILTER_REJECT
        : NodeFilter.FILTER_ACCEPT,
  });
  let words = 0;
  let node;
  while ((node = walker.nextNode())) {
    const t = node.nodeValue.trim();
    if (t) words += t.split(/\s+/).length;
  }
  wordCountEl.textContent = words + (words === 1 ? ' word' : ' words');
}
function setModelStatus(text, loading) {
  modelStatusEl.textContent = text;
  modelStatusEl.classList.toggle('loading', !!loading);
}
function setGhostStatus(text) {
  ghostStatusEl.textContent = text || '';
}

/* ── model state & worker ───────────────────────────────────── */
const worker = new Worker('worker.js');
const modelState = new Map(); // id -> { status, builtMaxN, stats }
let workerModelId = null; // model whose topology the worker currently holds

function stateOf(id) {
  if (!modelState.has(id)) modelState.set(id, { status: 'idle', builtMaxN: 0, stats: null });
  return modelState.get(id);
}

function ensureBuilt(id) {
  const m = modelById(id);
  const st = stateOf(id);
  const want = settings.maxNgram;
  if (st.status === 'ready' && st.builtMaxN >= want && workerModelId === id) return true;
  if (st.status === 'loading' && st.builtMaxN >= want) return false;

  if (id === 'abyss') {
    const corpus = abyssCorpus.value.trim();
    if (!corpus) {
      st.status = 'idle';
      setModelStatus('Abyss needs a corpus — paste one in Settings ⚙', false);
      return false;
    }
    localStorage.setItem('mllm.abyss', corpus);
    post({ cmd: 'build', modelId: id, corpusText: corpus, maxN: want });
  } else {
    post({ cmd: 'build', modelId: id, url: m.url, maxN: want });
  }
  st.status = 'loading';
  st.builtMaxN = want;
  setModelStatus('Loading ' + m.name + '…', true);
  return false;
}

function post(msg) {
  worker.postMessage(msg);
}

worker.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'phase') {
    const st = stateOf(msg.modelId);
    if (st.status === 'loading') {
      setModelStatus(
        modelById(msg.modelId).name + ': ' +
        (msg.phase === 'fetch' ? 'downloading corpus…' : 'building topology…'), true);
    }
    return;
  }
  if (msg.type === 'progress') {
    const st = stateOf(msg.modelId);
    if (st.status === 'loading') {
      setModelStatus(modelById(msg.modelId).name + ': ' + msg.sentences.toLocaleString() + ' sentences…', true);
    }
    return;
  }
  if (msg.type === 'built') {
    const st = stateOf(msg.modelId);
    st.status = 'ready';
    st.builtMaxN = msg.stats.maxN;
    st.stats = msg.stats;
    workerModelId = msg.modelId;
    if (settings.modelId === msg.modelId) {
      setModelStatus(
        modelById(msg.modelId).name + ' · ' +
        msg.stats.vocab.toLocaleString() + ' vocab · ' +
        msg.stats.tokens.toLocaleString() + ' tokens', false);
      scheduleTrigger(150);
    }
    return;
  }
  if (msg.type === 'error') {
    const st = stateOf(msg.modelId);
    st.status = 'error';
    st.builtMaxN = 0;
    if (settings.modelId === msg.modelId) {
      setModelStatus(modelById(msg.modelId).name + ' failed: ' + msg.message, false);
    }
    return;
  }
  if (msg.type === 'result') {
    if (msg.id !== gen) return; // stale
    if (!msg.continuation.length) return;
    renderGhost(msg.continuation, msg.confidences);
  }
};

/* ── model selection UI ─────────────────────────────────────── */
function buildModelUI() {
  for (const m of MODELS) {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = m.name + ' (' + m.size + ')';
    modelSelect.appendChild(opt);

    const card = document.createElement('button');
    card.className = 'model-card';
    card.dataset.model = m.id;
    card.innerHTML =
      '<span class="badge">' + m.size + '</span>' +
      '<span class="name">' + m.name + '</span><br>' +
      '<span class="meta">' + m.meta + '</span>';
    card.addEventListener('click', () => selectModel(m.id));
    modelCards.appendChild(card);
  }
  modelSelect.addEventListener('change', () => selectModel(modelSelect.value));
}

function selectModel(id, opts = {}) {
  settings.modelId = id;
  saveSettings();
  modelSelect.value = id;
  modelCards.querySelectorAll('.model-card').forEach((c) =>
    c.classList.toggle('selected', c.dataset.model === id));
  abyssBox.hidden = id !== 'abyss';
  dismissGhost();
  if (!opts.silent) {
    const st = stateOf(id);
    if (st.status === 'ready' && st.builtMaxN >= settings.maxNgram && st.stats && workerModelId === id) {
      setModelStatus(modelById(id).name + ' · ' + st.stats.vocab.toLocaleString() +
        ' vocab · ' + st.stats.tokens.toLocaleString() + ' tokens', false);
      scheduleTrigger(150);
    } else {
      ensureBuilt(id);
    }
  }
}

/* ── settings panel ─────────────────────────────────────────── */
document.getElementById('btnSettings').addEventListener('click', () => {
  settingsPanel.hidden = !settingsPanel.hidden;
  document.getElementById('btnSettings').classList.toggle('active', !settingsPanel.hidden);
});

function bindSlider(id, valId, key, fmt, after) {
  const el = document.getElementById(id);
  const val = document.getElementById(valId);
  el.value = settings[key];
  val.textContent = fmt(settings[key]);
  el.addEventListener('input', () => {
    settings[key] = parseFloat(el.value);
    val.textContent = fmt(settings[key]);
    saveSettings();
    dismissGhost();
    if (after) after();
    else scheduleTrigger(200);
  });
}
const fmtInt = (v) => String(v);
const fmtFloat = (v) => v.toFixed(2).replace(/0+$/, '').replace(/\.$/, '.0');

bindSlider('optSteps', 'valSteps', 'steps', fmtInt);
bindSlider('optTemp', 'valTemp', 'temperature', fmtFloat);
bindSlider('optThreshold', 'valThreshold', 'threshold', fmtFloat);
bindSlider('optMaxNgram', 'valMaxNgram', 'maxNgram', fmtInt, () => {
  // higher orders need a rebuild; lower ones reuse the topology
  ensureBuilt(settings.modelId);
});

const seedRandomEl = document.getElementById('optSeedRandom');
const seedEl = document.getElementById('optSeed');
seedRandomEl.checked = settings.seed == null;
seedEl.disabled = settings.seed == null;
if (settings.seed != null) seedEl.value = settings.seed;
seedRandomEl.addEventListener('change', () => {
  seedEl.disabled = seedRandomEl.checked;
  settings.seed = seedRandomEl.checked ? null : (parseInt(seedEl.value, 10) || 0);
  saveSettings();
  dismissGhost();
  scheduleTrigger(200);
});
seedEl.addEventListener('input', () => {
  if (seedRandomEl.checked) return;
  settings.seed = parseInt(seedEl.value, 10) || 0;
  saveSettings();
});

const autocompleteEl = document.getElementById('optAutocomplete');
autocompleteEl.checked = settings.autocomplete;
autocompleteEl.addEventListener('change', () => {
  settings.autocomplete = autocompleteEl.checked;
  saveSettings();
  if (!settings.autocomplete) dismissGhost();
  else scheduleTrigger(200);
});

/* Abyss BYO corpus */
abyssCorpus.value = localStorage.getItem('mllm.abyss') || ABYSS_SAMPLE;
document.getElementById('btnBuildAbyss').addEventListener('click', () => {
  const st = stateOf('abyss');
  st.status = 'idle';
  st.builtMaxN = 0;
  ensureBuilt('abyss');
});

/* ── ghost text ─────────────────────────────────────────────── */
let ghost = null; // { span, fullText }
let ghostCaret = null; // { node, offset }
let gen = 0;
let triggerTimer = null;

function textBeforeCaret() {
  const sel = document.getSelection();
  if (!sel.rangeCount) return '';
  const range = document.createRange();
  range.selectNodeContents(page);
  try {
    range.setEnd(sel.anchorNode, sel.anchorOffset);
  } catch (_) {
    return '';
  }
  return range.toString();
}

function dismissGhost() {
  gen++;
  if (ghost && ghost.span.parentNode) ghost.span.remove();
  ghost = null;
  ghostCaret = null;
  setGhostStatus('');
}

function scheduleTrigger(ms) {
  clearTimeout(triggerTimer);
  triggerTimer = setTimeout(() => triggerCompletion(false), ms == null ? 450 : ms);
}

function triggerCompletion(manual) {
  dismissGhost();
  if (!manual && !settings.autocomplete) return;
  if (document.activeElement !== page) return;
  const sel = document.getSelection();
  if (!sel.rangeCount || !sel.isCollapsed || !selectionInPage()) return;

  const st = stateOf(settings.modelId);
  if (st.status !== 'ready') {
    if (st.status === 'idle') ensureBuilt(settings.modelId);
    return;
  }
  const before = textBeforeCaret();
  const tokens = MLLM.tokenize(before.slice(-1200));
  if (!tokens.length) return;

  const id = ++gen;
  worker.postMessage({
    cmd: 'complete',
    id,
    modelId: settings.modelId,
    prefix: tokens,
    opts: {
      maxTokens: Math.round(settings.steps),
      temperature: settings.temperature,
      threshold: settings.threshold,
      maxN: Math.round(settings.maxNgram),
      seed: settings.seed,
    },
  });
}

// Fold sentence punctuation into the preceding token for display, mirroring
// assemble_autocomplete's tightening. Returns [{text, conf}].
function ghostSegments(tokens, confs) {
  const segs = [];
  tokens.forEach((tok, i) => {
    if (/^[.,!?]$/.test(tok) && segs.length) {
      segs[segs.length - 1].text += tok;
    } else {
      segs.push({ text: tok, conf: confs[i] != null ? confs[i] : 0 });
    }
  });
  return segs;
}

function renderGhost(tokens, confs) {
  const sel = document.getSelection();
  if (!sel.rangeCount || !sel.isCollapsed || !selectionInPage()) return;
  if (document.activeElement !== page) return;

  const before = textBeforeCaret();
  const endsSentence = before.trim() === '' || /[.!?]\s*$/.test(before);
  const segs = ghostSegments(tokens, confs);
  if (!segs.length) return;
  if (endsSentence && segs[0].text) {
    segs[0].text = segs[0].text[0].toUpperCase() + segs[0].text.slice(1);
  }
  let display = segs.map((s) => s.text).join(' ');
  const prevChar = before.slice(-1);
  if (prevChar && !/\s/.test(prevChar) && !/^[.,!?]/.test(display)) {
    display = ' ' + display;
    segs[0].text = ' ' + segs[0].text;
  }

  const span = document.createElement('span');
  span.className = 'ghost';
  span.contentEditable = 'false';
  for (const s of segs) {
    const t = document.createElement('span');
    t.className = 'gtok';
    t.textContent = s.text;
    t.style.opacity = (0.3 + 0.7 * Math.min(1, s.conf)).toFixed(2);
    span.appendChild(t);
    span.appendChild(document.createTextNode(' '));
  }
  span.removeChild(span.lastChild); // trailing space

  const range = sel.getRangeAt(0).cloneRange();
  range.collapse(true);
  range.insertNode(span);
  // put the caret back where it was — immediately before the ghost
  range.setStartBefore(span);
  range.collapse(true);
  sel.removeAllRanges();
  sel.addRange(range);

  ghost = { span, fullText: display };
  ghostCaret = { node: sel.anchorNode, offset: sel.anchorOffset };
  const avg = confs.length ? confs.reduce((a, b) => a + b, 0) / confs.length : 0;
  setGhostStatus('Tab to accept · Esc to dismiss · avg confidence ' + avg.toFixed(2));
}

function acceptGhost() {
  if (!ghost) return;
  const text = ghost.fullText;
  const span = ghost.span;
  // null out *before* execCommand: its beforeinput listener would otherwise
  // dismiss the ghost (removing the selected span) before the text lands.
  ghost = null;
  ghostCaret = null;
  const sel = document.getSelection();
  const range = document.createRange();
  range.selectNode(span);
  sel.removeAllRanges();
  sel.addRange(range);
  document.execCommand('insertText', false, text);
  setGhostStatus('');
  updateCounts();
  saveSoon();
  scheduleTrigger(350); // chain the next suggestion
}

/* ── editor events ──────────────────────────────────────────── */
page.addEventListener('beforeinput', () => dismissGhost());
page.addEventListener('input', () => {
  updateCounts();
  saveSoon();
  scheduleTrigger(450);
});

page.addEventListener('keydown', (e) => {
  if (e.key === 'Tab' && ghost) {
    e.preventDefault();
    acceptGhost();
    return;
  }
  if (e.key === 'Escape') {
    if (ghost) {
      e.preventDefault();
      dismissGhost();
    }
    return;
  }
  // any other key moves the caret / changes text → drop the suggestion
  if (ghost && !e.metaKey && !e.ctrlKey && !e.altKey) dismissGhost();
  else if (ghost && (e.key.startsWith('Arrow') || e.key === 'Backspace' || e.key === 'Delete')) dismissGhost();
});

document.addEventListener('keydown', (e) => {
  if (e.ctrlKey && !e.metaKey && e.code === 'Space') {
    e.preventDefault();
    triggerCompletion(true);
  }
});

document.addEventListener('selectionchange', () => {
  updateFmtStates();
  if (!ghost || !ghostCaret) return;
  const sel = document.getSelection();
  if (!sel.rangeCount || !sel.isCollapsed ||
      sel.anchorNode !== ghostCaret.node || sel.anchorOffset !== ghostCaret.offset) {
    dismissGhost();
  }
});

document.getElementById('btnComplete').addEventListener('click', () => {
  page.focus();
  triggerCompletion(true);
});

/* ── boot ───────────────────────────────────────────────────── */
buildModelUI();
updateCounts();
selectModel(settings.modelId);
page.focus();
