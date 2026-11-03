// LyricBridge AI: page logic. Whisper runs in worker.js; lyric maths lives in lyrics.js.
import {
  planWindows, chunksToRows, alignTranslations, fixTimings, buildSrt, buildVtt,
  rowsToText, activeRow, fmtClock
} from './lyrics.js';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const SAMPLE = {
  url: 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/french-audio.mp3',
  name: 'sample-french-speech.mp3'
};
const MAX_SECONDS = 10 * 60;
const BATCH = 40;
const displayNames = (() => { try { return new Intl.DisplayNames(['en'], { type: 'language' }); } catch { return null; } })();
const langName = (code) => (code && displayNames ? displayNames.of(code) : code) || 'Unknown';

// Example result: original lines written for this demo, not taken from a real song.
const EXAMPLE = [
  { start: 4.2, end: 8.6, orig: 'Bajo la luna te esperé', en: 'Under the moon I waited for you', note: '' },
  { start: 8.6, end: 13.4, orig: 'con el corazón en la mano', en: 'with my heart on my sleeve', note: 'Spanish idiom for speaking with complete sincerity, holding nothing back.' },
  { start: 13.4, end: 19.8, orig: 'y me importa un pepino el qué dirán', en: "and I couldn't care less what people say", note: "Literally 'a cucumber matters to me'; it means not caring at all." },
  { start: 19.8, end: 24.6, orig: 'contigo el tiempo vuela', en: 'with you, time flies', note: '' },
  { start: 24.6, end: 30.2, orig: 'eres mi media naranja', en: "you're my other half", note: "Literally 'my half orange', a common way to say soulmate." },
  { start: 30.2, end: 36.0, orig: 'y esta canción es nuestro puente', en: 'and this song is our bridge', note: '' }
];

const state = {
  rows: EXAMPLE.map(r => ({ ...r })),
  example: true,
  source: 'audio',
  file: null,
  audioUrl: null,
  audio: null,
  duration: 0,
  windows: [],
  language: '',
  detected: null,
  model: 'base',
  mode: 'whisper',
  llm: { available: false, provider: null, model: null, failed: '' },
  running: false,
  runId: 0,
  token: null,
  worker: null,
  engine: 'LLM, meaning first',
  editing: false,
  lastUserScroll: 0
};

/* ---------------- small UI helpers ---------------- */
let toastTimer;
function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 2600);
}
function setStep(name, status, detail) {
  const li = $(`#steps [data-step="${name}"]`);
  if (!li) return;
  li.dataset.status = status;
  if (detail !== undefined) $('.step-detail', li).textContent = detail;
}
function resetSteps() {
  $$('#steps li').forEach(li => { li.dataset.status = 'todo'; $('.step-detail', li).textContent = ''; });
}
function setProgress(fraction, msg) {
  $('#progBar').style.width = `${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}%`;
  if (msg) $('#progMsg').textContent = msg;
}
function setRunning(on) {
  state.running = on;
  $('#startBtn').hidden = on;
  $('#stopBtn').hidden = !on;
  $('#progress').dataset.active = String(on);
  $$('#studio .lock').forEach(el => { el.disabled = on; });
}
function mb(bytes) { return (bytes / 1048576).toFixed(1); }

/* ---------------- LLM availability ---------------- */
async function checkLlm() {
  const status = $('#llmStatus');
  try {
    const res = await fetch('/api/translate', { headers: { Accept: 'application/json' } });
    if (!res.ok) throw new Error(String(res.status));
    const info = await res.json();
    state.llm = { available: !!info.llm, provider: info.provider, model: info.model, failed: '' };
  } catch {
    state.llm = { available: false, provider: null, model: null, failed: '' };
  }
  const llmRadio = $('#modeLlm');
  if (state.llm.available) {
    llmRadio.disabled = false;
    status.textContent = `On: ${state.llm.provider} · ${state.llm.model}`;
    status.dataset.on = 'true';
    llmRadio.checked = true;
    state.mode = 'llm';
  } else {
    llmRadio.disabled = true;
    status.textContent = 'Not set up on this server, so Whisper translates on your device.';
    status.dataset.on = 'false';
    $('#modeWhisper').checked = true;
    state.mode = 'whisper';
  }
  $$('[data-llm-state]').forEach(el => { el.textContent = state.llm.available ? 'Connected' : 'Not set up'; });
}

function markLlmFailed(reason) {
  state.llm.failed = reason;
  const llmRadio = $('#modeLlm');
  llmRadio.disabled = true;
  $('#modeWhisper').checked = true;
  state.mode = 'whisper';
  $('#llmStatus').textContent = `Unavailable right now (${reason}). Whisper translates instead.`;
  $('#llmStatus').dataset.on = 'false';
}

/* ---------------- audio input ---------------- */
async function decodeTo16k(arrayBuffer) {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  let decoded;
  try { decoded = await ctx.decodeAudioData(arrayBuffer); } finally { ctx.close?.(); }
  const length = Math.max(1, Math.ceil(decoded.duration * 16000));
  const off = new OfflineAudioContext(1, length, 16000);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return { samples: rendered.getChannelData(0), duration: decoded.duration };
}

async function loadFile(file) {
  if (!file) return;
  if (state.running) stopRun();
  if (file.size > 80 * 1048576) { toast('That file is over 80 MB. Try a shorter clip or an MP3.'); return; }
  $('#fileName').textContent = `Reading ${file.name}…`;
  try {
    const { samples, duration } = await decodeTo16k(await file.arrayBuffer());
    if (duration > MAX_SECONDS) { toast('Songs up to 10 minutes work best. Trim it and try again.'); $('#fileName').textContent = 'No song yet'; return; }
    state.file = file;
    state.audio = samples;
    state.duration = duration;
    if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
    state.audioUrl = URL.createObjectURL(file);
    const audioEl = $('#audio');
    audioEl.src = state.audioUrl;
    $('#fileName').textContent = `${file.name} · ${fmtClock(duration)}`;
    $('#dropzone').dataset.loaded = 'true';
    $('#startBtn').disabled = false;
    $('#timeTotal').textContent = fmtClock(duration);
    toast('Song loaded. Press "Translate this song" when you are ready.');
  } catch (err) {
    console.error(err);
    $('#fileName').textContent = 'No song yet';
    toast("This browser can't read that file. MP3, WAV, M4A, OGG and WEBM usually work.");
  }
}

async function loadSample() {
  $('#fileName').textContent = 'Fetching the sample clip…';
  try {
    const res = await fetch(SAMPLE.url);
    if (!res.ok) throw new Error(String(res.status));
    const blob = await res.blob();
    await loadFile(new File([blob], SAMPLE.name, { type: blob.type || 'audio/mpeg' }));
  } catch {
    $('#fileName').textContent = 'No song yet';
    toast("Couldn't fetch the sample clip. Check your connection, or drop in your own song.");
  }
}

/* ---------------- worker runs ---------------- */
function getWorker() {
  if (!state.worker) {
    state.worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  }
  return state.worker;
}

function stopRun() {
  if (state.worker) { state.worker.terminate(); state.worker = null; }
  state.runId++;
  state.token = null;
  setRunning(false);
  setProgress(0, 'Stopped. Your lines so far are kept.');
  $$('#steps li[data-status="active"]').forEach(li => { li.dataset.status = 'todo'; });
}

/** Run the worker once. Resolves with { language } when done. */
function runWorker({ tasks, language, onWindow }) {
  return new Promise((resolve, reject) => {
    const id = ++state.runId;
    const worker = getWorker();
    const files = new Map();
    const copy = state.audio.slice();
    worker.onmessage = ({ data }) => {
      if (data.id !== id) return;
      if (data.type === 'load-progress') {
        if (data.file) files.set(data.file, { loaded: data.loaded, total: data.total });
        let loaded = 0, total = 0;
        for (const f of files.values()) { loaded += f.loaded; total += f.total; }
        if (total > 0) {
          setStep('model', 'active', `${mb(loaded)} of ${mb(total)} MB`);
          setProgress(loaded / total, `Downloading Whisper ${state.model} (only the first time, then it's cached)`);
        }
      } else if (data.type === 'ready') {
        setStep('model', 'done', data.device === 'webgpu' ? 'WebGPU' : 'CPU');
        $('#deviceBadge').textContent = data.device === 'webgpu' ? 'Running on WebGPU' : 'Running on CPU (WASM)';
        if (!language) setStep('lang', 'active', 'listening…');
      } else if (data.type === 'status') {
        $('#progMsg').textContent = data.message;
      } else if (data.type === 'detected') {
        state.detected = { code: data.code, confidence: data.confidence };
        setStep('lang', 'done', `${langName(data.code)} · ${Math.round(data.confidence * 100)}%`);
      } else if (data.type === 'window') {
        setStep('lyrics', 'active', `${fmtClock(data.end)} of ${fmtClock(state.duration)}`);
        setProgress((data.index + 1) / data.total, `Picking up lyrics: part ${data.index + 1} of ${data.total}`);
        onWindow(data);
      } else if (data.type === 'done') {
        resolve({ language: data.language });
      } else if (data.type === 'error') {
        reject(new Error(data.message));
      }
    };
    worker.onerror = (e) => reject(new Error(e.message || 'The worker stopped unexpectedly.'));
    worker.postMessage({ type: 'run', id, model: state.model, audio: copy, windows: state.windows, language, tasks }, [copy.buffer]);
  });
}

async function translateWithLlm(rows, language) {
  const todo = rows.filter(r => r.orig);
  for (let i = 0; i < todo.length; i += BATCH) {
    const batch = todo.slice(i, i + BATCH);
    setProgress(i / todo.length, `Translating with ${state.llm.provider}: lines ${i + 1}–${i + batch.length} of ${todo.length}`);
    const res = await fetch('/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ language: langName(language), lines: batch.map(r => r.orig) })
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.error || `server answered ${res.status}`);
    batch.forEach((r, k) => { r.en = body.lines?.[k]?.en || r.en; r.note = body.lines?.[k]?.note || ''; });
    renderRows();
  }
}

async function startAudioRun() {
  if (!state.audio) { toast('Drop in a song first, or try the sample clip.'); return; }
  state.example = false;
  state.rows = [];
  state.detected = null;
  state.windows = planWindows(state.audio);
  currentIndex = -1;
  const token = Symbol('run');
  state.token = token;
  renderRows();
  resetSteps();
  setRunning(true);
  setStep('model', 'active', 'loading…');
  setProgress(0, `Loading Whisper ${state.model}…`);
  const useLlm = state.mode === 'llm' && state.llm.available;
  state.engine = useLlm ? `LLM (${state.llm.provider}), meaning first` : 'Whisper translate, on your device';
  const tasks = useLlm ? ['transcribe'] : ['transcribe', 'translate'];
  const language = state.language || null;
  if (language) setStep('lang', 'done', langName(language));
  try {
    const { language: lang } = await runWorker({
      tasks, language,
      onWindow: (w) => {
        const rows = chunksToRows(w.results.transcribe, w.start, w.end);
        if (w.results.translate) {
          const tr = w.results.translate.map(c => ({ start: w.start + (c.start ?? 0), end: c.end == null ? w.end : w.start + c.end, text: c.text }));
          alignTranslations(rows, tr);
        }
        state.rows.push(...rows);
        renderRows();
      }
    });
    if (state.token !== token) return;
    fixTimings(state.rows, state.duration);
    const resolved = lang || language || 'en';
    setStep('lyrics', 'done', `${state.rows.length} lines`);
    if (!state.rows.length) {
      setStep('translate', 'skip', 'nothing to translate');
      finish('No singing found. Try a bigger model or pick the language yourself.');
      return;
    }
    if (resolved === 'en') {
      state.rows.forEach(r => { r.en = ''; });
      setStep('translate', 'skip', 'already English');
      finish('This song is already in English, so the lyrics are ready to export.');
      return;
    }
    setStep('translate', 'active', useLlm ? state.llm.provider : 'Whisper');
    if (useLlm) {
      try {
        await translateWithLlm(state.rows, resolved);
      } catch (err) {
        markLlmFailed(String(err.message || err).slice(0, 80));
        setStep('translate', 'active', 'falling back to Whisper');
        state.engine = 'Whisper translate, on your device';
        await runWorker({
          tasks: ['translate'], language: resolved,
          onWindow: (w) => {
            const inWin = state.rows.filter(r => r.start >= w.start - 0.01 && r.start < w.end);
            const tr = (w.results.translate || []).map(c => ({ start: w.start + (c.start ?? 0), end: c.end == null ? w.end : w.start + c.end, text: c.text }));
            inWin.forEach(r => { r.en = ''; });
            alignTranslations(inWin, tr);
            renderRows();
          }
        });
      }
    }
    if (state.token !== token) return;
    setStep('translate', 'done', state.engine.startsWith('LLM') ? 'meaning first' : 'Whisper');
    finish('Done. Press play to follow along, edit any line, then export subtitles.');
  } catch (err) {
    if (state.token !== token) return;
    console.error(err);
    const msg = /memory|allocation|OOM/i.test(err.message) ? 'Your device ran out of memory. Try the Tiny or Base model.' : `Something went wrong: ${err.message}`;
    $$('#steps li[data-status="active"]').forEach(li => { li.dataset.status = 'error'; });
    setRunning(false);
    setProgress(0, msg);
    toast(msg);
  }
}

async function startTextRun() {
  const lines = $('#textInput').value.split('\n').map(s => s.trim()).filter(Boolean);
  if (!lines.length) { toast('Paste a few lines of lyrics first.'); return; }
  if (!state.llm.available) { toast('Pasted lyrics need the LLM, which is not set up on this server. Songs work on your device.'); return; }
  if (lines.length > 200) { toast('Paste at most 200 lines at a time.'); return; }
  state.example = false;
  currentIndex = -1;
  state.rows = lines.map(orig => ({ start: NaN, end: NaN, orig, en: '', note: '' }));
  state.engine = `LLM (${state.llm.provider}), meaning first`;
  renderRows();
  resetSteps();
  setRunning(true);
  setStep('model', 'skip', 'not needed');
  setStep('lyrics', 'done', `${lines.length} lines pasted`);
  const lang = state.language || '';
  setStep('lang', lang ? 'done' : 'skip', lang ? langName(lang) : 'the LLM works it out');
  setStep('translate', 'active', state.llm.provider);
  try {
    await translateWithLlm(state.rows, lang);
    setStep('translate', 'done', 'meaning first');
    finish('Done. Pasted lyrics have no timings, so subtitles need a song file.');
  } catch (err) {
    markLlmFailed(String(err.message || err).slice(0, 80));
    $$('#steps li[data-status="active"]').forEach(li => { li.dataset.status = 'error'; });
    setRunning(false);
    setProgress(0, 'The LLM could not translate right now. Try a song file instead.');
  }
}

function finish(message) {
  setRunning(false);
  setProgress(1, message);
  renderRows();
  toast(message);
}

/* ---------------- lyrics view ---------------- */
function rowTime(r) { return Number.isFinite(r.start) ? fmtClock(r.start) : '—'; }

function renderRows() {
  const list = $('#lyrics');
  const rows = state.rows;
  $('#exampleBadge').hidden = !state.example;
  const timed = rows.length && rows.every(r => Number.isFinite(r.start));
  $$('[data-needs-timing]').forEach(b => { b.disabled = !timed; });
  $$('[data-needs-rows]').forEach(b => { b.disabled = !rows.length; });
  const lang = state.example ? 'Spanish' : langName(state.detected?.code || state.language) ;
  $('#resultMeta').textContent = rows.length
    ? `${rows.length} lines · ${state.example ? 'example' : lang} · ${state.example ? 'LLM translation' : state.engine}`
    : (state.running ? 'Lines appear here as Whisper hears them.' : 'No lines yet.');
  list.innerHTML = '';
  if (!rows.length) {
    const li = document.createElement('li');
    li.className = 'empty';
    li.textContent = state.running ? 'Listening…' : 'Your lyrics will show up here, line by line.';
    list.appendChild(li);
    updateNow(-1);
    return;
  }
  const frag = document.createDocumentFragment();
  rows.forEach((r, i) => {
    const li = document.createElement('li');
    li.className = 'line';
    li.dataset.i = String(i);
    const time = document.createElement('button');
    time.type = 'button';
    time.className = 'line-time';
    time.textContent = rowTime(r);
    time.disabled = !Number.isFinite(r.start) || state.example;
    time.setAttribute('aria-label', `Play from ${rowTime(r)}`);
    time.addEventListener('click', () => seekTo(r.start, true));
    const body = document.createElement('div');
    body.className = 'line-body';
    const orig = document.createElement('p');
    orig.className = 'line-orig';
    orig.textContent = r.orig;
    orig.dataset.field = 'orig';
    const en = document.createElement('p');
    en.className = 'line-en';
    en.textContent = r.en || (state.running ? '…' : '');
    en.dataset.field = 'en';
    body.append(orig, en);
    if (r.note) {
      const note = document.createElement('p');
      note.className = 'line-note';
      note.textContent = r.note;
      note.dataset.field = 'note';
      body.append(note);
    }
    if (state.editing) {
      $$('[data-field]', body).forEach(el => {
        el.contentEditable = 'true';
        el.spellcheck = false;
        el.addEventListener('blur', () => { state.rows[i][el.dataset.field] = el.textContent.trim(); updateNow(currentIndex); });
      });
    }
    li.append(time, body);
    frag.appendChild(li);
  });
  list.appendChild(frag);
  const nowIdx = state.example ? 2 : currentIndex;
  $$('#lyrics .line').forEach(li => li.classList.toggle('is-now', Number(li.dataset.i) === nowIdx));
  updateNow(state.example ? 2 : Math.max(0, currentIndex));
}

let currentIndex = -1;
function updateNow(i) {
  const r = state.rows[i];
  $('#nowOrig').textContent = r ? r.orig : (state.running ? 'Listening…' : 'Drop in a song to begin');
  $('#nowEn').textContent = r ? (r.en || '') : 'The English line appears here as it plays.';
  $('#nowNote').textContent = r?.note || '';
  $('#nowNote').hidden = !r?.note;
}

function highlight(i, scroll) {
  if (i === currentIndex) return;
  currentIndex = i;
  $$('#lyrics .line').forEach(li => li.classList.toggle('is-now', Number(li.dataset.i) === i));
  updateNow(i);
  if (scroll && i >= 0 && Date.now() - state.lastUserScroll > 2500) {
    const list = $('#lyricsScroll');
    const li = $(`#lyrics .line[data-i="${i}"]`);
    if (li) list.scrollTo({ top: li.offsetTop - list.clientHeight / 3, behavior: reduceMotion ? 'auto' : 'smooth' });
  }
}

/* ---------------- player ---------------- */
function seekTo(t, play) {
  const a = $('#audio');
  if (!state.audioUrl) return;
  a.currentTime = t;
  if (play) a.play();
}

function wirePlayer() {
  const a = $('#audio');
  const seek = $('#seek');
  const play = $('#playBtn');
  play.addEventListener('click', () => {
    if (!state.audioUrl) { toast('Load a song to play it here.'); return; }
    if (a.paused) a.play(); else a.pause();
  });
  a.addEventListener('play', () => { play.dataset.playing = 'true'; play.setAttribute('aria-label', 'Pause'); $('#nowVinyl').dataset.spin = 'true'; });
  a.addEventListener('pause', () => { play.dataset.playing = 'false'; play.setAttribute('aria-label', 'Play'); $('#nowVinyl').dataset.spin = 'false'; });
  a.addEventListener('loadedmetadata', () => { seek.max = String(a.duration || 0); $('#timeTotal').textContent = fmtClock(a.duration || 0); });
  a.addEventListener('timeupdate', () => {
    if (!seek.matches(':active')) seek.value = String(a.currentTime);
    $('#timeNow').textContent = fmtClock(a.currentTime);
    if (!state.example) highlight(activeRow(state.rows, a.currentTime), true);
  });
  seek.addEventListener('input', () => { a.currentTime = Number(seek.value); });
  $('#lyricsScroll').addEventListener('wheel', () => { state.lastUserScroll = Date.now(); }, { passive: true });
  $('#lyricsScroll').addEventListener('touchmove', () => { state.lastUserScroll = Date.now(); }, { passive: true });
}

/* ---------------- export ---------------- */
function baseName() {
  if (state.example) return 'lyricbridge-example';
  return (state.file?.name || 'lyrics').replace(/\.[^.]+$/, '').replace(/[^\w.-]+/g, '-').slice(0, 60) || 'lyrics';
}
function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement('a'), { href: url, download: name });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
  toast(`Saved ${name}`);
}
function wireExport() {
  $('#exportSrtBoth').addEventListener('click', () => download(`${baseName()}.bilingual.srt`, buildSrt(state.rows, 'both'), 'application/x-subrip'));
  $('#exportSrtEn').addEventListener('click', () => download(`${baseName()}.en.srt`, buildSrt(state.rows, 'en'), 'application/x-subrip'));
  $('#exportVtt').addEventListener('click', () => download(`${baseName()}.bilingual.vtt`, buildVtt(state.rows, 'both'), 'text/vtt'));
  $('#copyBtn').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(rowsToText(state.rows)); toast('Lyrics copied'); }
    catch { toast('Copy was blocked by the browser. Select the lines and copy them instead.'); }
  });
  $('#editToggle').addEventListener('click', () => {
    state.editing = !state.editing;
    $('#editToggle').setAttribute('aria-pressed', String(state.editing));
    $('#editToggle').textContent = state.editing ? 'Done editing' : 'Edit lines';
    renderRows();
  });
  $('#exampleBtn').addEventListener('click', () => {
    if (state.running) return;
    state.example = true;
    state.rows = EXAMPLE.map(r => ({ ...r }));
    renderRows();
  });
}

/* ---------------- controls ---------------- */
function wireControls() {
  const input = $('#fileInput');
  const drop = $('#dropzone');
  input.addEventListener('change', () => loadFile(input.files[0]));
  ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.dataset.over = 'true'; }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.dataset.over = 'false'; }));
  drop.addEventListener('drop', (e) => loadFile(e.dataTransfer.files[0]));
  $('#sampleBtn').addEventListener('click', loadSample);

  $$('[data-src]').forEach(tab => tab.addEventListener('click', () => {
    state.source = tab.dataset.src;
    $$('[data-src]').forEach(t => t.setAttribute('aria-selected', String(t === tab)));
    $('#panelAudio').hidden = state.source !== 'audio';
    $('#panelText').hidden = state.source !== 'text';
    $('#startBtn').disabled = state.source === 'audio' ? !state.audio : false;
    $('#startBtn').querySelector('span').textContent = state.source === 'audio' ? 'Translate this song' : 'Translate these lyrics';
  }));

  $('#langSelect').addEventListener('change', (e) => { state.language = e.target.value; });
  $$('input[name="model"]').forEach(r => r.addEventListener('change', () => { if (r.checked) state.model = r.value; }));
  $$('input[name="mode"]').forEach(r => r.addEventListener('change', () => { if (r.checked) state.mode = r.value; }));
  $('#startBtn').addEventListener('click', () => (state.source === 'audio' ? startAudioRun() : startTextRun()));
  $('#stopBtn').addEventListener('click', stopRun);

  $$('[data-use-model]').forEach(b => b.addEventListener('click', () => {
    const r = $(`input[name="model"][value="${b.dataset.useModel}"]`);
    if (r && !r.disabled) { r.checked = true; state.model = r.value; }
    $('#studio').scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth' });
    toast(`Whisper ${b.dataset.useModel} selected`);
  }));
  $$('[data-use-llm]').forEach(b => b.addEventListener('click', () => {
    if (!state.llm.available) { toast('The LLM is not set up on this server yet.'); return; }
    $('#modeLlm').checked = true; state.mode = 'llm';
    $('#studio').scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth' });
  }));
}

/* ---------------- "how it works" scroll story ---------------- */
function wireStory() {
  const story = $('#how');
  const steps = $$('.story-step', story);
  const pills = $$('.story-pill', story);
  const desktop = matchMedia('(min-width: 900px)');
  const set = (i) => {
    steps.forEach((s, k) => s.classList.toggle('is-on', k === i));
    pills.forEach((p, k) => p.setAttribute('aria-current', String(k === i)));
  };
  const onScroll = () => {
    if (!desktop.matches || reduceMotion) return;
    const r = story.getBoundingClientRect();
    const span = r.height - innerHeight;
    const p = Math.min(0.999, Math.max(0, -r.top / Math.max(1, span)));
    set(Math.floor(p * steps.length));
  };
  pills.forEach((p, k) => p.addEventListener('click', () => {
    if (!desktop.matches || reduceMotion) { steps[k].scrollIntoView({ behavior: 'auto', block: 'center' }); set(k); return; }
    const top = story.offsetTop + (story.offsetHeight - innerHeight) * (k + 0.5) / steps.length;
    scrollTo({ top, behavior: 'smooth' });
  }));
  document.documentElement.classList.add('story-js');
  set(0);
  addEventListener('scroll', onScroll, { passive: true });
  addEventListener('resize', onScroll);
  onScroll();
}

/* ---------------- start ---------------- */
function init() {
  $('#deviceBadge').textContent = navigator.gpu ? 'WebGPU ready' : 'CPU mode (WASM)';
  wireControls();
  wirePlayer();
  wireExport();
  wireStory();
  renderRows();
  checkLlm();
  $$('[data-year]').forEach(el => { el.textContent = String(new Date().getFullYear()); });
}
init();
