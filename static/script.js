// ── DOM refs ─────────────────────────────────────────────────────────────
const inputText      = document.getElementById('inputText');
const summarizeBtn   = document.getElementById('summarizeBtn');
const clearBtn       = document.getElementById('clearBtn');
const copyBtn        = document.getElementById('copyBtn');
const charCount      = document.getElementById('charCount');

const outputPlaceholder = document.getElementById('outputPlaceholder');
const outputLoading     = document.getElementById('outputLoading');
const outputResult      = document.getElementById('outputResult');
const outputError       = document.getElementById('outputError');
const summaryText       = document.getElementById('summaryText');
const errorText         = document.getElementById('errorText');
const statsBar          = document.getElementById('statsBar');
const statWords         = document.getElementById('statWords');
const statSumWords      = document.getElementById('statSumWords');
const statRatio         = document.getElementById('statRatio');

// ── State ─────────────────────────────────────────────────────────────────
let currentSummary = '';

// ── Helpers ───────────────────────────────────────────────────────────────
const wordCount = (str) => str.trim().split(/\s+/).filter(Boolean).length;

function showState(state) {
  outputPlaceholder.hidden = state !== 'idle';
  outputLoading.hidden     = state !== 'loading';
  outputResult.hidden      = state !== 'result';
  outputError.hidden       = state !== 'error';
}

function updateCharCount() {
  const n = inputText.value.length;
  charCount.textContent = `${n.toLocaleString()} character${n !== 1 ? 's' : ''}`;
}

// ── Summarise ──────────────────────────────────────────────────────────────
async function summarize() {
  const text = inputText.value.trim();
  if (!text) {
    showError('Please enter some text before summarising.');
    return;
  }

  // UI: loading
  showState('loading');
  summarizeBtn.classList.add('loading');
  summarizeBtn.disabled = true;
  copyBtn.disabled = true;
  statsBar.hidden = true;

  try {
    const resp = await fetch('/summarize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    const data = await resp.json();

    if (!resp.ok) {
      // Flask returns {error}, FastAPI returns {detail} (a string, or a list for validation errors).
      const detail = Array.isArray(data.detail) ? data.detail[0]?.msg : data.detail;
      throw new Error(data.error || detail || `Server error ${resp.status}`);
    }

    currentSummary = data.summary;
    summaryText.textContent = currentSummary;
    showState('result');
    copyBtn.disabled = false;

    // Stats
    const inWords  = wordCount(text);
    const outWords = wordCount(currentSummary);
    const ratio    = inWords > 0 ? Math.round((1 - outWords / inWords) * 100) : 0;
    statWords.textContent    = `${inWords.toLocaleString()} words`;
    statSumWords.textContent = `${outWords.toLocaleString()} words`;
    statRatio.textContent    = `${ratio}% shorter`;
    statsBar.hidden = false;

  } catch (err) {
    showError(err.message || 'Something went wrong. Is the server running?');
  } finally {
    summarizeBtn.classList.remove('loading');
    summarizeBtn.disabled = false;
  }
}

function showError(msg) {
  errorText.textContent = msg;
  showState('error');
  statsBar.hidden = true;
}

// ── Copy ──────────────────────────────────────────────────────────────────
async function copySummary() {
  if (!currentSummary) return;
  try {
    await navigator.clipboard.writeText(currentSummary);
    copyBtn.innerHTML = `
      <svg viewBox="0 0 20 20" fill="currentColor">
        <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/>
      </svg>`;
    copyBtn.style.borderColor = '#22c55e';
    copyBtn.style.color = '#22c55e';
    setTimeout(() => {
      copyBtn.innerHTML = `
        <svg viewBox="0 0 20 20" fill="currentColor">
          <path d="M8 2a2 2 0 00-2 2v10a2 2 0 002 2h6a2 2 0 002-2V7.414A2 2 0 0015.414 6L12 2.586A2 2 0 0010.586 2H8zM6 4a2 2 0 012-2h2.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V14a2 2 0 01-2 2H8a2 2 0 01-2-2V4z"/>
          <path d="M4 6a2 2 0 00-2 2v8a2 2 0 002 2h6a2 2 0 002-2v-1H8a4 4 0 01-4-4V6z"/>
        </svg>`;
      copyBtn.style.borderColor = '';
      copyBtn.style.color = '';
    }, 2000);
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
  updateCharCount();
  currentSummary = '';
  copyBtn.disabled = true;
  statsBar.hidden = true;
  showState('idle');
  inputText.focus();
}

// ── Events ────────────────────────────────────────────────────────────────
summarizeBtn.addEventListener('click', summarize);
clearBtn.addEventListener('click', clearInput);
copyBtn.addEventListener('click', copySummary);

inputText.addEventListener('input', updateCharCount);

// Keyboard shortcut: Ctrl/Cmd + Enter
inputText.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') summarize();
});

// ── Init ──────────────────────────────────────────────────────────────────
updateCharCount();
showState('idle');
