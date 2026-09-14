# Cloudflare Workers AI — Benchmark

Benchmarks **Cloudflare Workers AI** across text, image, and audio generation over 56 prompts, then compares the scores against Claude, ChatGPT, and Gemini.

---

## Folder Structure

```
cloudflare/
├── cloudflare_benchmark.py      # Main script: generate + score all 56 prompts
├── cf_vs_bigplayers.py          # Comparison graphs: Cloudflare vs big players
├── score_audio_only.py          # Standalone audio scorer (CLAP + MOS + WER)
├── cloudflare_visualizations.ipynb  # Jupyter notebook: per-prompt score plots
├── test_cloudflare_all.py       # Quick API test (single prompt, all 3 modalities)
├── prompts.py                   # 56 benchmark prompts
└── outputs/
    ├── text/                    # prompt_1.txt … prompt_56.txt
    ├── images/                  # prompt_1.png … prompt_56.png
    ├── audio/                   # prompt_1.wav … prompt_56.wav
    ├── scores/
    │   ├── text_scores.csv      # BERTScore, ROUGE-L, Readability per prompt
    │   ├── image_scores.csv     # CLIP, Aesthetic per prompt
    │   └── audio_scores.csv     # CLAP, MOS, WER per prompt
    └── graphs/
        ├── text/                # 5 charts (bar + grouped + line)
        ├── image/               # 4 charts (bar + grouped + line)
        ├── audio/               # 5 charts (bar + grouped + line)
        └── cf_vs_big_combined.png
```

---

## Models Used

| Modality | Cloudflare Model |
|----------|-----------------|
| Text | `@cf/meta/llama-3.1-8b-instruct` |
| Image | `@cf/black-forest-labs/flux-1-schnell` |
| Audio | `@cf/myshell-ai/melotts` |

---

## Scoring Metrics

| Modality | Metric | Description | Range |
|----------|--------|-------------|-------|
| Text | **BERTScore** | Semantic similarity to prompt | 0–1 (higher = better) |
| Text | **ROUGE-L** | Lexical overlap with prompt | 0–1 (higher = better) |
| Text | **Readability** | Flesch-Kincaid grade / 10 | 0–∞ (context-dependent) |
| Image | **CLIP** | Image-text alignment (ViT-L/14) | 0–1 (higher = better) |
| Image | **Aesthetic** | Visual quality (LAION predictor) | 0–10 (higher = better) |
| Audio | **CLAP** | Audio-text alignment | 0–1 (higher = better) |
| Audio | **MOS** | Speech quality (DNSMOS) | 1–5 (higher = better) |
| Audio | **WER** | Word error rate (Whisper) | 0–1 (lower = better) |

---

## How to Run

### Prerequisites

```bash
pip install torch transformers bert-score rouge-score textstat
pip install openai-whisper jiwer soundfile scipy librosa
pip install speechmos onnxruntime Pillow requests matplotlib
```

Make sure `config.py` exists in `testing multiple models/` with your Cloudflare credentials:

```python
CLOUDFLARE = {
    "account_id": "your_account_id",
    "api_token":  "your_api_token",
}
```

---

### Step 1 — Generate + Score Everything

```bash
python cloudflare_benchmark.py
```

- Generates text, image, and audio for all 56 prompts (skips already-generated files)
- Scores all outputs and saves CSVs to `outputs/scores/`
- On quota exhaustion (HTTP 429) it stops generation and scores what exists

---

### Step 2 — Score Audio Only (if needed separately)

```bash
python score_audio_only.py
```

Runs CLAP + MOS + WER without re-generating. Useful if audio scoring failed in the main run. Does not require `ffmpeg` — uses `soundfile` for audio loading.

---

### Step 3 — Generate Comparison Graphs

```bash
python cf_vs_bigplayers.py
```

Generates 15 charts comparing Cloudflare vs Claude / ChatGPT / Gemini, saved into `outputs/graphs/text/`, `image/`, `audio/`.

---

### Step 4 — Per-Prompt Visualizations (optional)

Open `cloudflare_visualizations.ipynb` in Jupyter and run all cells. Shows per-prompt score distributions for all metrics.

---

### Quick API Test

```bash
python test_cloudflare_all.py
```

Sends one prompt to all 3 Cloudflare models and saves sample outputs to `outputs/cf_test/`. Use this to verify your API credentials work.

---

## Benchmark Results (Averages over 56 prompts)

| Metric | Cloudflare | Claude | ChatGPT | Gemini |
|--------|-----------|--------|---------|--------|
| BERTScore | 0.8371 | 0.851 | 0.832 | 0.796 |
| ROUGE-L | 0.0557 | 0.078 | 0.071 | 0.068 |
| Readability | 0.9480 | 1.85 | 1.72 | 1.68 |
| CLIP | — | 0.312 | 0.305 | 0.298 |
| Aesthetic | — | 7.20 | 7.05 | 6.90 |
| CLAP | 0.0822 | 0.74 | 0.72 | 0.76 |
| MOS | 3.367 | 4.10 | 4.05 | 4.20 |

---

## Notes

- Generation is **cached** — re-running skips already-generated files.
- CLAP model (~2GB) and CLIP model (~1.7GB) download automatically on first run.
- `aesthetic_weights.pth` (~3.7MB) is downloaded once for image aesthetic scoring.
- WER = 1.0 is expected — TTS speaks the full generated paragraph, not the short prompt.
