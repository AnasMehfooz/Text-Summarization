// ── DOM refs ─────────────────────────────────────────────────────────────
const inputText      = document.getElementById('inputText');
const summarizeBtn   = document.getElementById('summarizeBtn');
const clearBtn       = document.getElementById('clearBtn');
const copyBtn        = document.getElementById('copyBtn');
const copyLabel      = document.getElementById('copyLabel');
const retryBtn       = document.getElementById('retryBtn');
const charCount      = document.getElementById('charCount');
const formatHint     = document.getElementById('formatHint');
const shortcutHint   = document.getElementById('shortcutHint');
const statusEl       = document.getElementById('status');
const statusText     = document.getElementById('statusText');

const outputPlaceholder = document.getElementById('outputPlaceholder');
const outputLoading     = document.getElementById('outputLoading');
const outputResult      = document.getElementById('outputResult');
const outputError       = document.getElementById('outputError');
const summaryText       = document.getElementById('summaryText');
const errorTitle        = document.getElementById('errorTitle');
const errorText         = document.getElementById('errorText');
const statsBar          = document.getElementById('statsBar');
const statWords         = document.getElementById('statWords');
const statSumWords      = document.getElementById('statSumWords');
const statRatio         = document.getElementById('statRatio');
const statTime          = document.getElementById('statTime');

// ── Examples ──────────────────────────────────────────────────────────────
const EXAMPLES = {
  hike: `Maya: Are we still on for the hike on Saturday?
Leo: Yes! But can we start a bit later, like 10?
Maya: Sure. I'll bring sandwiches and water.
Leo: Great, I'll drive. I'll pick you up at 9:30.`,
  keys: `Sam: Have you seen my car keys? I can't find them anywhere.
Priya: Did you check the jacket you wore yesterday?
Sam: Not there either...
Priya: Maybe you left them at the gym?
Sam: Oh no, probably. I'll call them right now.`,
  gift: `Nora: It's Dad's birthday on Friday and I still have no gift.
Omar: He keeps talking about that new coffee grinder.
Nora: Good idea! Want to split it?
Omar: Sure, send me the link and I'll pay half.`,
};

// ── State ─────────────────────────────────────────────────────────────────
let currentSummary = '';
let isLoading = false;

// ── Helpers ───────────────────────────────────────────────────────────────
const wordCount = (str) => str.trim().split(/\s+/).filter(Boolean).length;

// A chat is at least two "Name: message" lines.
const looksLikeDialogue = (str) =>
  (str.match(/^\s*[^:\n]{1,40}:\s*\S/gm) || []).length >= 2;

function showState(state) {
  outputPlaceholder.hidden = state !== 'idle';
  outputLoading.hidden     = state !== 'loading';
  outputResult.hidden      = state !== 'result';
  outputError.hidden       = state !== 'error';
}

function updateInputState() {
  const text  = inputText.value;
  const words = wordCount(text);
  const chars = text.length;
  charCount.textContent = chars
    ? `${words.toLocaleString()} word${words !== 1 ? 's' : ''} · ${chars.toLocaleString()} characters`
    : '0 words';

  const hasText = text.trim().length > 0;
  summarizeBtn.disabled = !hasText || isLoading;
  clearBtn.disabled     = !hasText;
  formatHint.hidden     = text.trim().length < 40 || looksLikeDialogue(text);
}

function setStatus(state, label) {
  statusEl.dataset.state = state;
  statusText.textContent = label;
}

// The model returns all-lowercase text. Restore casing for names and other
// words that are capitalised mid-sentence in the input, then sentence-case it.
function restoreCase(summary, source) {
  const forms = new Map();   // lowercase word → { form: count }
  const bump = (word) => {
    const key = word.toLowerCase();
    const counts = forms.get(key) || {};
    counts[word] = (counts[word] || 0) + 1;
    forms.set(key, counts);
  };

  for (const m of source.matchAll(/[A-Za-z]+/g)) {
    const before = source.slice(0, m.index).trimEnd();
    const sentenceStart = before === '' || /[.!?:\n]$/.test(before);
    if (!sentenceStart) bump(m[0]);
  }
  // Speaker labels ("Hannah:") are names even though they start a line.
  for (const m of source.matchAll(/^\s*([^:\n]{1,40}):/gm)) {
    for (const w of m[1].match(/[A-Za-z]+/g) || []) bump(w);
  }

  const best = new Map();
  for (const [key, counts] of forms) {
    const [form] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
    if (form !== key) best.set(key, form);
  }

  return summary
    .replace(/[A-Za-z]+/g, (w) => (w === 'i' ? 'I' : best.get(w) || w))
    .replace(/(^\s*|[.!?]\s+)([a-z])/g, (m, pre, ch, offset, str) =>
      // Skip abbreviations like "a.m." or "p.m."
      /(^|[^A-Za-z])[A-Za-z]\.$/.test(str.slice(0, offset + 1)) ? m : pre + ch.toUpperCase());
}

// ── Summarise ──────────────────────────────────────────────────────────────
async function summarize() {
  const text = inputText.value.trim();
  if (!text || isLoading) return;

  // UI: loading
  isLoading = true;
  showState('loading');
  summarizeBtn.classList.add('loading');
  updateInputState();
  copyBtn.disabled = true;
  statsBar.hidden = true;

  const started = performance.now();
  try {
    let resp;
    try {
      resp = await fetch('/summarize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
    } catch {
      setStatus('offline', 'Server offline');
      showError("Can't reach the server", 'Start it with "python app.py" in the project folder, then try again.');
      return;
    }

    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      // Flask returns {error}, FastAPI returns {detail} (a string, or a list for validation errors).
      const detail = Array.isArray(data.detail) ? data.detail[0]?.msg : data.detail;
      showError('Summarization failed', data.error || detail || `The server responded with error ${resp.status}.`);
      return;
    }

    const seconds = (performance.now() - started) / 1000;
    currentSummary = restoreCase(data.summary, text);
    summaryText.textContent = currentSummary;
    showState('result');
    copyBtn.disabled = false;

    // Stats
    const inWords  = wordCount(text);
    const outWords = wordCount(currentSummary);
    const ratio    = inWords > 0 ? Math.round((1 - outWords / inWords) * 100) : 0;
    statWords.textContent    = inWords.toLocaleString();
    statSumWords.textContent = outWords.toLocaleString();
    statRatio.textContent    = ratio >= 0 ? `${ratio}% shorter` : 'Longer than input';
    statTime.textContent     = `${seconds.toFixed(1)}s`;
    statsBar.hidden = false;
  } finally {
    isLoading = false;
    summarizeBtn.classList.remove('loading');
    updateInputState();
  }
}

function showError(title, msg) {
  errorTitle.textContent = title;
  errorText.textContent  = msg;
  showState('error');
  statsBar.hidden = true;
}

// ── Health ────────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const resp = await fetch('/health', { cache: 'no-store' });
    const data = await resp.json();
    if (!resp.ok) throw new Error();
    if (data.status === 'loading') {
      setStatus('checking', 'Loading model…');
      return;
    }
    setStatus('ready', `Ready · ${String(data.device).startsWith('cuda') ? 'GPU' : 'CPU'}`);
  } catch {
    setStatus('offline', 'Server offline');
  }
}

// ── Copy ──────────────────────────────────────────────────────────────────
async function copySummary() {
  if (!currentSummary) return;
  try {
    await navigator.clipboard.writeText(currentSummary);
    copyBtn.classList.add('copied');
    copyLabel.textContent = 'Copied';
    setTimeout(() => {
      copyBtn.classList.remove('copied');
      copyLabel.textContent = 'Copy';
    }, 1800);
  } catch {
    // fallback — select text
    const range = document.createRange();
    range.selectNodeContents(summaryText);
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(range);
  }
}

// ── Clear ─────────────────────────────────────────────────────────────────
function clearInput() {
  inputText.value = '';
  currentSummary = '';
  copyBtn.disabled = true;
  statsBar.hidden = true;
  showState('idle');
  updateInputState();
  inputText.focus();
}

// ── Events ────────────────────────────────────────────────────────────────
summarizeBtn.addEventListener('click', summarize);
retryBtn.addEventListener('click', () => { checkHealth(); summarize(); });
clearBtn.addEventListener('click', clearInput);
copyBtn.addEventListener('click', copySummary);

inputText.addEventListener('input', updateInputState);

document.querySelectorAll('[data-example]').forEach((chip) => {
  chip.addEventListener('click', () => {
    inputText.value = EXAMPLES[chip.dataset.example];
    updateInputState();
    summarize();
  });
});

// Keyboard shortcut: Ctrl/Cmd + Enter
inputText.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') summarize();
});

window.addEventListener('focus', checkHealth);

// ── Init ──────────────────────────────────────────────────────────────────
if (/Mac|iPhone|iPad/.test(navigator.platform)) shortcutHint.textContent = '⌘ ↵';
updateInputState();
showState('idle');
checkHealth();
setInterval(checkHealth, 15000);
