# LyricBridge AI

**Song lyrics in any language, translated into English for what they mean.**

Drop in a song. LyricBridge picks up the lyrics with OpenAI's Whisper running **in your browser**, translates them into English with their meaning intact, plays the song with synced bilingual lyrics, and exports subtitles (SRT and VTT).

**Live:** https://lyricbridge-ai.vercel.app

## Features

| | |
| --- | --- |
| **Lyrics from the song** | Whisper (tiny, base or small) transcribes the vocals with timestamps, on WebGPU when available and on the CPU (WebAssembly) otherwise. |
| **Language detection** | Scores Whisper's language tokens on three slices of the song, so an instrumental intro doesn't fool it. Or pick the language yourself. |
| **Meaning-first translation** | An LLM translates each line for meaning and adds a short note for idioms and slang ("me importa un pepino" becomes "I couldn't care less"). Without a key, Whisper translates speech to English on the device. |
| **Synced player** | Play the song and the current line lights up; click any timestamp to jump there. |
| **Edit, then export** | Fix any line by hand, then download bilingual or English SRT, or bilingual WebVTT. Copy the lyric sheet as text. |
| **Private by design** | Audio is decoded and transcribed in the browser and never uploaded. Only lyric text goes to the LLM, and only when you use it. |

## How it works

```
song file ──► Web Audio API: decode, mix to mono, resample to 16 kHz
          ──► split into ≤30 s windows, cut at the quietest moment (no sliced words), skip silent parts
          ──► Web Worker: Whisper via Transformers.js (WebGPU or WASM, model cached after first download)
                 ├─ language detection (if not chosen)
                 ├─ task "transcribe" ──► original lines + timestamps
                 └─ task "translate"  ──► English (on-device mode), aligned to lines by time overlap
          ──► /api/translate (Python on Vercel) ──► LLM: meaning-first English + idiom notes
          ──► synced player · editable lyric sheet · SRT / VTT export
```

If the LLM call fails (no key, quota, network), the app falls back to Whisper's on-device translation automatically.

## Tech stack

- **Frontend:** HTML, CSS and vanilla JavaScript (ES modules), Web Audio API, Web Workers
- **Speech:** [Whisper](https://github.com/openai/whisper) ONNX models from `onnx-community` on Hugging Face, run with [Transformers.js](https://github.com/huggingface/transformers.js) 3.8.1
- **Translation API:** Python serverless function on Vercel (`api/translate.py`, standard library only)
- **LLMs:** Gemini, OpenAI, Groq or Vercel AI Gateway, whichever credential is configured
- **Hosting:** Vercel

## Turn on the LLM translation

Add one of these environment variables to the Vercel project (Settings → Environment Variables), then redeploy:

| Variable | Provider | Default model |
| --- | --- | --- |
| `GEMINI_API_KEY` | Google Gemini (free key from Google AI Studio) | `gemini-flash-lite-latest` |
| `OPENAI_API_KEY` | OpenAI | `gpt-4.1-mini` |
| `GROQ_API_KEY` | Groq | `llama-3.3-70b-versatile` |
| `AI_GATEWAY_API_KEY` | Vercel AI Gateway (the deployment's OIDC token also works) | `google/gemini-2.5-flash-lite` |

Optional: `LLM_PROVIDER` to pick one explicitly, `LLM_MODEL` to change the model, `LLM_DISABLED=1` to switch the endpoint off.
`GET /api/translate` reports what the server is using.

## Run locally

```bash
python3 -m http.server 8000   # the page and on-device Whisper (no LLM)
vercel dev                    # the page plus the Python API
npm test                      # JavaScript and Python unit tests
```

## Project structure

```
index.html        page markup (Chispa-inspired design)
styles.css        all styles
app.js            page logic: audio decoding, runs, player, editing, export
worker.js         Whisper in a Web Worker, language detection
lyrics.js         pure helpers: windows, alignment, timing fixes, SRT/VTT
api/translate.py  LLM translation endpoint (Vercel Python function)
tests/            node:test and unittest suites
```

## Good to know

- Accuracy drops when the music is much louder than the voice. The Small model hears through music best.
- On a CPU, a full song can take a few minutes with Base. WebGPU is several times faster.
- The example lyrics on the page were written for this demo and are not from a real song.

## Credits

Whisper by OpenAI (MIT). Transformers.js by Hugging Face (Apache-2.0). Design inspired by the Chispa restaurant template.

Built by **Vedant Gadage**: [portfolio](https://vedant-gadage.vercel.app) · [GitHub](https://github.com/Vedantsg23)

## License

MIT
