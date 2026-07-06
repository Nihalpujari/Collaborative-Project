# Cloudflare Workers AI — Team Requirements

Owner: Anuj (Cloudflare API & Requirements)
Last verified: live-tested against the real API, not just docs.

This doc tells each teammate exactly what the Cloudflare API returns, so
you can build your part without re-discovering these details yourself.

---

## Credentials (already set up)

`config.py` (gitignored, in repo root of `testing multiple models/`) has:

```python
CLOUDFLARE = {
    "account_id": "...",
    "api_token":  "...",
}
```

Base URL pattern:
```
https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}
Header: Authorization: Bearer {api_token}
```

Ask Anuj if you need the live values — don't create your own token, it
splits our shared 10,000 neurons/day free quota across accounts (see
`LIMITS.md`).

---

## Models used (confirmed working)

| Modality | Model ID | Status |
|---|---|---|
| Text | `@cf/meta/llama-3.1-8b-instruct` | ✅ Reliable |
| Image | `@cf/black-forest-labs/flux-1-schnell` | ✅ Reliable |
| Audio | `@cf/myshell-ai/melotts` | ⚠️ Reliable but ~25% intermittent 500 errors — see below |

---

## For Namrata (Multimodal Pipeline)

Request/response shapes, verified live:

**Text — POST body:**
```json
{"messages": [{"role": "user", "content": "your prompt"}], "max_tokens": 200}
```
**Text — response:**
```json
{"result": {"response": "generated text..."}, "success": true}
```
Extract with `r.json()["result"]["response"]`.

**Image — POST body:**
```json
{"prompt": "your prompt"}
```
**Image — response:**
```json
{"result": {"image": "<base64 string>"}, "success": true}
```
Decode with `base64.b64decode(...)`.
**Note:** despite being commonly saved as `.png`, the actual bytes returned
are **JPEG** (magic bytes `FF D8 FF`, not `89 50 4E 47`). This doesn't
break anything (PIL/CLIP auto-detect format from bytes, not extension) but
don't assume PNG-specific behavior (e.g. transparency) — there isn't any.

**Audio — POST body:**
```json
{"prompt": "text to speak"}
```
**Audio — response:**
```json
{"result": {"audio": "<base64 WAV string>"}, "success": true}
```
Decode with `base64.b64decode(...)` → raw WAV file bytes, ready to write
to disk as-is.

**Pipeline order that makes sense:** generate text first → feed that text
into audio generation (so the TTS speaks the actual generated content, not
the raw prompt) → generate image from the original prompt independently.
This is what `cloudflare_benchmark.py` already does.

---

## For Gourav (CLIP Scoring — Image)

- Cloudflare images are **JPEG bytes**, not PNG, despite common naming —
  `PIL.Image.open()` handles this fine automatically, just don't hardcode
  format assumptions.
- Default resolution from `flux-1-schnell` with a plain `{"prompt": ...}`
  call (no explicit width/height) — confirm actual dimensions with `PIL`
  before assuming 512×512 or 1024×1024; we did not pass explicit size
  params in our test call, so the model used its own default. Check
  `im.size` per image rather than assuming a fixed constant, in case it
  varies.
- All 56 images will be in `cloudflare/outputs/images/prompt_1.png` …
  `prompt_56.png` (named `.png` in the repo convention even though the
  underlying bytes are JPEG — this is fine, just know why `imghdr`/magic
  bytes checks might surprise you).

---

## For Pramod (CLAP Scoring — Audio)

**Correction to your task description:** the audio does **not** come back
at 48kHz. Verified directly from the WAV header:

| Property | Actual value |
|---|---|
| Sample rate | **44,100 Hz** (not 48kHz) |
| Channels | 1 (mono) |
| Sample width | 16-bit PCM |
| Format | Standard WAV container, decodable with `wave`, `soundfile`, or `librosa` directly |

Whatever resampling step you build, resample **from 44.1kHz**, not 48kHz,
to whatever target rate CLAP/MOS/WER models expect (CLAP typically wants
48kHz, Whisper wants 16kHz — so you likely need *two different* resample
targets depending on which scorer you're feeding, not one universal
48kHz conversion).

**Reliability warning:** `@cf/myshell-ai/melotts` intermittently returns
`HTTP 500 Internal Server Error` — confirmed by sending the **exact same
request 8 times**, which failed 2 out of 8 times with no pattern related
to prompt content or length. This is a Cloudflare-side issue, not
something wrong with your code or the payload.

`cloudflare_benchmark.py`'s `generate_audio()` has been updated with a
3-attempt retry + exponential backoff for this. If you're scoring audio
independently via `score_audio_only.py`, make sure any *missing* files in
`outputs/audio/` are because of true generation failure after retries, not
because the script gave up after one 500. If you see gaps in your
`prompt_N.wav` sequence, re-run generation for just those IDs rather than
assuming the prompt itself is bad.

---

## For Nihal (Frontend / Visualization)

CSV output columns you'll be reading from `outputs/scores/`:
- `text_scores.csv` — BERTScore, ROUGE-L, Readability, per prompt ID
- `image_scores.csv` — CLIP, Aesthetic, per prompt ID
- `audio_scores.csv` — CLAP, MOS, WER, per prompt ID

Comparison baselines (Claude/ChatGPT/Gemini) are hardcoded in
`config.py` → `BIG_PLAYERS_BENCHMARKS` dict, sourced from published 2025
papers (see comment block above that dict for arXiv links) — not
re-generated, so don't expect matching output files for those, just the
static numbers for your charts.

---

## Known issues log (update this as we find more)

| Issue | Status | Fix |
|---|---|---|
| `melotts` ~25% intermittent 500 errors | Fixed | Retry w/ backoff added to `generate_audio()` in `cloudflare_benchmark.py` |
| Image bytes are JPEG despite `.png` naming | Documented | No fix needed — PIL handles it; just don't assume PNG-specific features |
| Audio sample rate is 44.1kHz, not 48kHz as originally assumed in task description | Documented | Pramod should resample from 44.1kHz to each scorer's actual required rate |
