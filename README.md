# AI Model Benchmarking Suite

A comprehensive benchmarking framework that evaluates AI models across **text**, **image**, and **audio** generation — comparing smaller/API-based models against big players like Claude, ChatGPT, and Gemini.

---

## Project Structure

```
Collaborative-Project/
├── testing multiple models/
│   ├── cloudflare/          # Cloudflare Workers AI benchmark
│   ├── other models/        # Mistral, Groq, Cerebras benchmarks
│   ├── config.py            # API keys (gitignored — do not commit)
│   ├── config.example.py    # Template for config.py
│   └── README.md
├── .gitignore
└── README.md                ← you are here
```

---

## What It Does

| Modality | Generation | Scoring Metrics |
|----------|-----------|-----------------|
| **Text** | LLM text generation | BERTScore, ROUGE-L, Flesch-Kincaid Readability |
| **Image** | Text-to-image | CLIP (image-text alignment), Aesthetic score (0–10) |
| **Audio** | Text-to-speech | CLAP (audio-text alignment), MOS (speech quality), WER |

All models are benchmarked over **56 prompts** covering creative writing, factual questions, descriptive scenes, and more.

---

## Models Benchmarked

### Small / API Models
| Model | Provider | Modality |
|-------|----------|----------|
| Llama 3.1 8B | Cloudflare Workers AI | Text, Image, Audio |
| Flux 1 Schnell | Cloudflare Workers AI | Image |
| MeloTTS | Cloudflare Workers AI | Audio |
| Llama 3.3 70B | Groq | Text |
| Mistral Large | Mistral AI | Text |
| GPT-OSS 120B | Cerebras | Text |

### Big Players (Reference Benchmarks)
| Model | Text | Image | Audio |
|-------|------|-------|-------|
| Claude 3.5 Sonnet | ✓ | ✓ | ✓ |
| GPT-4o / DALL-E 3 | ✓ | ✓ | ✓ |
| Gemini 2.5 / Imagen 3 | ✓ | ✓ | ✓ |

> Big player scores are sourced from published benchmarks (2025) and normalized to the same scoring methodology.

---

## Quick Start

### 1. Clone & Setup

```bash
git clone <repo-url>
cd "Collaborative-Project/testing multiple models"
pip install -r requirements.txt   # if available
```

### 2. Configure API Keys

```bash
cp config.example.py config.py
# Edit config.py and fill in your API keys
```

### 3. Run Benchmarks

**Cloudflare:**
```bash
cd cloudflare
python cloudflare_benchmark.py   # generate + score all 56 prompts
python cf_vs_bigplayers.py       # generate comparison graphs
```

**Other Models (Mistral / Groq / Cerebras):**
```bash
cd "other models"
python main.py                   # full pipeline
```

---

## Outputs

```
cloudflare/outputs/
├── text/              # Generated text files
├── images/            # Generated images
├── audio/             # Generated audio (WAV)
├── scores/
│   ├── text_scores.csv
│   ├── image_scores.csv
│   └── audio_scores.csv
└── graphs/
    ├── text/          # BERTScore, ROUGE, Readability charts
    ├── image/         # CLIP, Aesthetic charts
    ├── audio/         # CLAP, MOS, WER charts
    └── cf_vs_big_combined.png
```

---

## Setup Notes

- `config.py` is **gitignored** — never committed. Copy from `config.example.py`.
- Model weights (`.pth`) and generated outputs are also gitignored (large files).
- Audio scoring requires `ffmpeg` for WER OR use `score_audio_only.py` which bypasses ffmpeg via soundfile.
- CLAP model requires ~2GB download on first run.
- CLIP model requires ~1.7GB download on first run.

---

## Dependencies

```bash
pip install torch transformers bert-score rouge-score textstat
pip install openai-whisper jiwer soundfile scipy librosa
pip install speechmos onnxruntime
pip install Pillow requests matplotlib numpy pandas
```
