// LyricBridge AI: runs Whisper in a Web Worker so the page stays responsive.
// Transformers.js loads the ONNX model from the Hugging Face Hub once and caches it in the browser.
import { pipeline, env, Tensor } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowLocalModels = false;

const MODELS = {
  tiny: 'onnx-community/whisper-tiny',
  base: 'onnx-community/whisper-base',
  small: 'onnx-community/whisper-small'
};
const SR = 16000;
let loaded = { key: null, asr: null, device: null };

async function webgpuAvailable() {
  try {
    if (!self.navigator?.gpu) return false;
    return !!(await self.navigator.gpu.requestAdapter());
  } catch { return false; }
}

async function load(key, id) {
  if (loaded.key === key && loaded.asr) return loaded;
  if (loaded.asr) { try { await loaded.asr.dispose(); } catch { /* ignore */ } }
  const progress_callback = (p) => {
    if (p.status === 'progress' || p.status === 'done' || p.status === 'initiate') {
      postMessage({ type: 'load-progress', id, file: p.file, status: p.status, loaded: p.loaded || 0, total: p.total || 0 });
    }
  };
  let device = (await webgpuAvailable()) ? 'webgpu' : 'wasm';
  let asr;
  try {
    asr = await pipeline('automatic-speech-recognition', MODELS[key], {
      device,
      dtype: device === 'webgpu' ? { encoder_model: 'fp32', decoder_model_merged: 'q4' } : 'q8',
      progress_callback
    });
  } catch (err) {
    if (device !== 'webgpu') throw err;
    postMessage({ type: 'status', id, message: 'WebGPU did not start, switching to the CPU.' });
    device = 'wasm';
    asr = await pipeline('automatic-speech-recognition', MODELS[key], { device, dtype: 'q8', progress_callback });
  }
  loaded = { key, asr, device };
  return loaded;
}

async function tensorData(t) {
  if (t.location && t.location !== 'cpu' && t.ort_tensor?.getData) return await t.ort_tensor.getData();
  return t.data;
}

// Transformers.js 3.x does not detect the language itself, so score Whisper's language tokens
// directly: one decoder step after <|startoftranscript|>, on up to three 30 s slices of the song.
async function detectLanguage(asr, audio) {
  const gc = asr.model.generation_config;
  const langs = Object.entries(gc.lang_to_id || {});
  if (!langs.length) return { code: 'en', confidence: 0 };
  const sot = gc.decoder_start_token_id;
  const W = 30 * SR;
  const starts = audio.length <= W ? [0] : [0.2, 0.45, 0.7].map(p => Math.min(audio.length - W, Math.floor(audio.length * p)));
  const totals = new Map();
  for (const s of starts) {
    const { input_features } = await asr.processor(audio.subarray(s, s + W));
    const out = await asr.model({
      input_features,
      decoder_input_ids: new Tensor('int64', BigInt64Array.from([BigInt(sot)]), [1, 1])
    });
    const logits = await tensorData(out.logits);
    for (const t of Object.values(out.past_key_values || {})) t?.dispose?.();
    const vocab = out.logits.dims.at(-1);
    const base = logits.length - vocab;
    const vals = langs.map(([, tokenId]) => logits[base + tokenId]);
    const max = Math.max(...vals);
    const exps = vals.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    langs.forEach(([tok], i) => totals.set(tok, (totals.get(tok) || 0) + exps[i] / sum));
  }
  let bestTok = 'en', best = -1;
  for (const [tok, score] of totals) if (score > best) { best = score; bestTok = tok; }
  return { code: bestTok.replace(/[<|>]/g, ''), confidence: best / starts.length };
}

self.onmessage = async ({ data }) => {
  if (data.type !== 'run') return;
  const { id, model, audio, windows, language, tasks } = data;
  try {
    const { asr, device } = await load(model, id);
    postMessage({ type: 'ready', id, device });

    let lang = language;
    if (!lang) {
      postMessage({ type: 'status', id, message: 'Listening for the language…' });
      const det = await detectLanguage(asr, audio);
      lang = det.code;
      postMessage({ type: 'detected', id, ...det });
    }

    for (let i = 0; i < windows.length; i++) {
      const w = windows[i];
      const results = {};
      if (!w.silent) {
        const seg = audio.subarray(Math.floor(w.start * SR), Math.floor(w.end * SR));
        for (const task of tasks) {
          if (task === 'translate' && lang === 'en') continue; // already English
          const out = await asr(seg, { language: lang, task, return_timestamps: true });
          results[task] = (out.chunks || []).map(c => ({ start: c.timestamp[0], end: c.timestamp[1], text: c.text }));
        }
      }
      postMessage({ type: 'window', id, index: i, total: windows.length, start: w.start, end: w.end, results });
    }
    postMessage({ type: 'done', id, language: lang, device });
  } catch (err) {
    postMessage({ type: 'error', id, message: String(err?.message || err) });
  }
};
