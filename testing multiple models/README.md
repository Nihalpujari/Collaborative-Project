# Testing Multiple Models

Root of the benchmarking suite. Contains all model benchmarks, shared config, and prompts.

---

## Folder Structure

```
testing multiple models/
├── cloudflare/             # Cloudflare Workers AI benchmark
│   └── README.md
├── other models/           # Mistral, Groq, Cerebras benchmark
│   └── README.md
├── config.py               # API keys — GITIGNORED, never commit
├── config.example.py       # Template — copy to config.py and fill keys
└── README.md               ← you are here
```

---

## Config Setup

All API keys live in one file: `config.py`. It is listed in `.gitignore` and will never be pushed to GitHub.

```bash
# First-time setup
cp config.example.py config.py
# Then open config.py and fill in your actual API keys
```

`config.py` structure:

```python
CLOUDFLARE   = { "account_id": "...", "api_token": "..." }
TEXT_MODELS  = { "cerebras": {...}, "groq": {...}, "mistral": {...} }
IMAGE_MODELS = { "black_forest_labs": {...}, "stability_ai": {...}, ... }
AUDIO_MODELS = { "gemini_tts": {...}, "deepgram": {...} }
```

---

## Which Folder to Use?

| Goal | Go to |
|------|-------|
| Benchmark Cloudflare Workers AI | [`cloudflare/`](cloudflare/README.md) |
| Benchmark Mistral / Groq / Cerebras | [`other models/`](other%20models/README.md) |

---

## Shared Prompts

Both benchmarks use the same 56 prompts from `prompts.py` (copied into each subfolder). The prompts cover:
- Creative writing & storytelling
- Descriptive scenes (used for image generation)
- Factual and instructional content
- Conversational and Q&A style

---

## Big Player Reference Benchmarks

Scores for Claude, ChatGPT, and Gemini are sourced from published 2025 benchmarks and hardcoded in `config.py` under `BIG_PLAYERS_BENCHMARKS`. These use the **same scoring methodology** as the small models so comparisons are fair.

| Source | Metrics |
|--------|---------|
| Enterprise LLM Eval Benchmark 2025 (arXiv) | BERTScore, ROUGE |
| Human & MLLM Image Preference 2025 (arXiv) | CLIP, Aesthetic |
| EmergentTTS-Eval 2025 (arXiv) | WER |
| PromptLayer GPT-4o-Mini-TTS report 2025 | MOS |
