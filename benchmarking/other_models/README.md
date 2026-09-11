# Other Models — Benchmark (Mistral / Groq / Cerebras)

Benchmarks **Mistral**, **Groq (Llama 3.3)**, and **Cerebras** across text, image, and audio generation over 56 prompts.

---

## Folder Structure

```
other models/
├── main.py               # Full pipeline orchestrator (generate → score → graph)
├── generate_text.py      # Text generation (Cerebras, Groq, Mistral)
├── generate_image.py     # Image generation (Black Forest Labs, Stability AI, HuggingFace, Pixazo)
├── generate_audio.py     # Audio generation (Deepgram, Camb AI, Cartesia, Fish Audio)
├── score_text.py         # Text scoring (BERTScore, ROUGE-L, Readability)
├── score_image.py        # Image scoring (CLIP, Aesthetic)
├── score_audio.py        # Audio scoring (CLAP, MOS, WER)
├── graph.py              # Comparison graph generator
├── text_benchmark.py     # Standalone text-only benchmark
├── prompts.py            # 56 benchmark prompts
└── outputs/
    ├── text/             # Generated text per model per prompt
    ├── images/           # Generated images per model per prompt
    ├── audio/            # Generated audio per model per prompt
    ├── scores/
    │   ├── text_scores.csv
    │   ├── image_scores.csv
    │   └── audio_scores.csv
    └── graphs/           # All comparison charts
```

---

## Models Used

### Text Generation
| Model | Provider | API |
|-------|----------|-----|
| GPT-OSS 120B | Cerebras | `api.cerebras.ai` |
| Llama 3.3 70B Versatile | Groq | `api.groq.com` |
| Mistral Large Latest | Mistral AI | `api.mistral.ai` |

### Image Generation
| Model | Provider |
|-------|----------|
| Flux Pro | Black Forest Labs |
| Stable Diffusion XL | Stability AI |
| Stable Diffusion v1.5 | Hugging Face |
| Flux | Pixazo |

### Audio Generation (TTS)
| Model | Provider |
|-------|----------|
| Aura Asteria | Deepgram |
| Camb AI TTS | Camb AI |
| Cartesia TTS | Cartesia |
| Fish Audio TTS | Fish Audio |

---

## Scoring Metrics

| Modality | Metric | Description | Range |
|----------|--------|-------------|-------|
| Text | **BERTScore** | Semantic similarity (roberta-large) | 0–1 ↑ |
| Text | **ROUGE-L** | Lexical overlap with prompt | 0–1 ↑ |
| Text | **Readability** | Flesch-Kincaid grade / 10 | higher = more complex |
| Image | **CLIP** | Image-text alignment (ViT-L/14) | 0–1 ↑ |
| Image | **Aesthetic** | Visual quality (LAION predictor) | 0–10 ↑ |
| Audio | **CLAP** | Audio-text alignment | 0–1 ↑ |
| Audio | **MOS** | Speech quality (DNSMOS) | 1–5 ↑ |
| Audio | **WER** | Word error rate (Whisper) | 0–1 ↓ |

---

## How to Run

### Prerequisites

```bash
pip install torch transformers bert-score rouge-score textstat
pip install openai-whisper jiwer soundfile scipy librosa
pip install speechmos onnxruntime Pillow requests matplotlib
```

Ensure `config.py` in `testing multiple models/` contains your API keys:

```python
TEXT_MODELS  = { "cerebras": {...}, "groq": {...}, "mistral": {...} }
IMAGE_MODELS = { "black_forest_labs": {...}, ... }
AUDIO_MODELS = { "deepgram": {...}, ... }
```

---

### Run Full Pipeline

```bash
python main.py
```

Runs the complete pipeline in order:
1. Generate text for all models × 56 prompts
2. Generate images for all models × 56 prompts
3. Generate audio from generated text
4. Score text, image, audio outputs
5. Save CSVs to `outputs/scores/`
6. Generate comparison graphs to `outputs/graphs/`

---

### Text-Only Benchmark

```bash
python text_benchmark.py
```

Runs only text generation and scoring. Faster for text-focused experiments.

---

### Individual Scripts

```bash
python generate_text.py    # generate text only
python generate_image.py   # generate images only
python generate_audio.py   # generate audio only
python score_text.py       # score text outputs
python score_image.py      # score image outputs
python score_audio.py      # score audio outputs
python graph.py            # generate graphs from existing CSVs
```

---

## Notes

- API keys are read from `../config.py` (gitignored). Copy from `../config.example.py`.
- Outputs are gitignored to avoid committing large binary files.
- Each script checks for existing outputs and skips regeneration (cached).
