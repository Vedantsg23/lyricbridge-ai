// LyricBridge AI: pure helpers for audio windows, lyric rows, alignment and subtitles.
// No DOM and no network, so they run the same in the browser and under `node --test`.

export const SAMPLE_RATE = 16000;

/** Root-mean-square energy of consecutive frames (default 50 ms at 16 kHz). */
export function frameRms(samples, frame = 800) {
  const n = Math.max(1, Math.ceil(samples.length / frame));
  const out = new Float32Array(n);
  for (let f = 0; f < n; f++) {
    const a = f * frame;
    const b = Math.min(samples.length, a + frame);
    let sum = 0;
    for (let i = a; i < b; i++) sum += samples[i] * samples[i];
    out[f] = Math.sqrt(sum / Math.max(1, b - a));
  }
  return out;
}

/**
 * Split audio into windows Whisper can take (30 s at most). Each cut is placed at the quietest
 * 50 ms frame between `minLen` and `maxLen` seconds, so words are rarely sliced in half.
 * Windows whose average energy is near zero are marked `silent` and can be skipped.
 */
export function planWindows(samples, { sr = SAMPLE_RATE, maxLen = 30, minLen = 18, frame = 800, silence = 0.004 } = {}) {
  const total = samples.length / sr;
  if (total === 0) return [];
  const rms = frameRms(samples, frame);
  const frameSec = frame / sr;
  const windows = [];
  let start = 0;
  while (total - start > maxLen) {
    const lo = Math.floor((start + minLen) / frameSec);
    const hi = Math.min(rms.length - 1, Math.floor((start + maxLen) / frameSec) - 1);
    let best = lo;
    for (let f = lo; f <= hi; f++) if (rms[f] < rms[best]) best = f;
    const cut = Math.min(start + maxLen, (best + 0.5) * frameSec);
    windows.push({ start, end: cut });
    start = cut;
  }
  windows.push({ start, end: total });
  return windows.map(w => {
    const a = Math.floor(w.start / frameSec);
    const b = Math.max(a + 1, Math.ceil(w.end / frameSec));
    let sum = 0;
    for (let f = a; f < b && f < rms.length; f++) sum += rms[f];
    const mean = sum / (b - a);
    return { start: round3(w.start), end: round3(w.end), silent: mean < silence };
  });
}

const NOISE = [
  /^[\s\p{P}\p{S}]*$/u,                 // only punctuation, music notes or symbols
  /thank(s| you) for watching/i,
  /please subscribe|like and subscribe/i,
  /amara\.org|subtitles by|captions by/i,
  /sous-titr/i, /untertitel/i,
  /ご視聴ありがとうございました/,
  /^\s*[[(]\s*(music|instrumental|applause|laughter|silence)\s*[\])]\s*$/i
];

/** Whisper sometimes "hears" stock phrases over instrumentals; drop those. */
export function isNoise(text) {
  const t = (text || '').trim();
  return NOISE.some(re => re.test(t));
}

export function cleanText(text) {
  return (text || '').replace(/\s+/g, ' ').replace(/^[\s♪♫]+|[\s♪♫]+$/g, '').trim();
}

/**
 * Turn Whisper chunks for one window into lyric rows with absolute times.
 * `chunks` are `{ start, end, text }` relative to the window; `end` may be null.
 */
export function chunksToRows(chunks, offset, windowEnd) {
  const rows = [];
  for (const c of chunks || []) {
    const text = cleanText(c.text);
    if (!text || isNoise(text)) continue;
    const start = offset + (Number.isFinite(c.start) ? c.start : 0);
    let end = Number.isFinite(c.end) ? offset + c.end : windowEnd;
    if (end > windowEnd) end = windowEnd;
    const prev = rows[rows.length - 1];
    if (prev && prev.orig === text && start - prev.end < 0.3) { prev.end = Math.max(prev.end, end); continue; } // repetition loop
    rows.push({ start: round3(start), end: round3(end), orig: text, en: '', note: '' });
  }
  return rows;
}

function overlap(a, b) {
  return Math.max(0, Math.min(a.end, b.end) - Math.max(a.start, b.start));
}

/**
 * Attach translated chunks (Whisper's "translate" task, timed independently) to the
 * transcribed rows: each chunk goes to the row it overlaps most, or the nearest row.
 */
export function alignTranslations(rows, translated) {
  if (!rows.length) return rows;
  const buckets = rows.map(() => []);
  for (const t of translated) {
    const text = cleanText(t.text);
    if (!text || isNoise(text)) continue;
    let best = -1, bestOv = 0;
    rows.forEach((r, i) => { const ov = overlap(r, t); if (ov > bestOv) { bestOv = ov; best = i; } });
    if (best === -1) {
      const mid = (t.start + t.end) / 2;
      let bestD = Infinity;
      rows.forEach((r, i) => { const d = Math.abs((r.start + r.end) / 2 - mid); if (d < bestD) { bestD = d; best = i; } });
    }
    buckets[best].push(text);
  }
  rows.forEach((r, i) => { if (buckets[i].length) r.en = buckets[i].join(' '); });
  return rows;
}

/** Sort rows, make every cue last at least a moment, and stop cues from overlapping. */
export function fixTimings(rows, total = Infinity) {
  rows.sort((a, b) => a.start - b.start);
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    const next = rows[i + 1];
    if (!(r.end > r.start)) r.end = r.start + Math.max(1.2, Math.min(6, (r.orig || '').length / 14));
    if (next && r.end > next.start) r.end = Math.max(r.start + 0.4, next.start);
    if (r.end > total) r.end = total;
    r.start = round3(r.start); r.end = round3(r.end);
  }
  return rows;
}

export function fmtStamp(sec, sep = ',') {
  const ms = Math.max(0, Math.round(sec * 1000));
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const r = ms % 1000;
  return `${pad(h)}:${pad(m)}:${pad(s)}${sep}${String(r).padStart(3, '0')}`;
}

export function fmtClock(sec) {
  const s = Math.max(0, Math.floor(sec));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function cueText(r, mode) {
  if (mode === 'orig') return r.orig;
  if (mode === 'en') return r.en || r.orig;
  return r.en && r.en !== r.orig ? `${r.orig}\n${r.en}` : r.orig;
}

/** SubRip subtitles. mode: 'en' | 'orig' | 'both' */
export function buildSrt(rows, mode = 'both') {
  return rows
    .filter(r => Number.isFinite(r.start) && Number.isFinite(r.end))
    .map((r, i) => `${i + 1}\n${fmtStamp(r.start)} --> ${fmtStamp(r.end)}\n${cueText(r, mode)}\n`)
    .join('\n');
}

/** WebVTT subtitles. mode: 'en' | 'orig' | 'both' */
export function buildVtt(rows, mode = 'both') {
  const cues = rows
    .filter(r => Number.isFinite(r.start) && Number.isFinite(r.end))
    .map(r => `${fmtStamp(r.start, '.')} --> ${fmtStamp(r.end, '.')}\n${cueText(r, mode)}\n`);
  return `WEBVTT\n\n${cues.join('\n')}`;
}

/** Plain bilingual lyric sheet for copying. */
export function rowsToText(rows) {
  return rows.map(r => {
    const time = Number.isFinite(r.start) ? `[${fmtClock(r.start)}] ` : '';
    const en = r.en && r.en !== r.orig ? `\n    ${r.en}` : '';
    const note = r.note ? `\n    (${r.note})` : '';
    return `${time}${r.orig}${en}${note}`;
  }).join('\n');
}

/** Index of the row playing at `t` seconds, or -1. Rows must be sorted by start. */
export function activeRow(rows, t) {
  let lo = 0, hi = rows.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (rows[mid].start <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  if (ans >= 0 && t > rows[ans].end + 1.5) return -1;
  return ans;
}

function pad(n) { return String(n).padStart(2, '0'); }
function round3(x) { return Math.round(x * 1000) / 1000; }
