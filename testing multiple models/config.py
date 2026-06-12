"""
Configuration file for all API keys and model settings.
Copy this file and fill in your actual API keys.
This file is safe to commit — it contains only placeholders.

To add your real keys, create a local override:
    my_keys.py  (gitignored)
Or edit the values below directly (config.py is gitignored after first setup).
"""

# ============================================================================
# CLOUDFLARE WORKERS AI
# ============================================================================
CLOUDFLARE = {
    "account_id": "YOUR_CLOUDFLARE_ACCOUNT_ID",
    "api_token":  "YOUR_CLOUDFLARE_API_TOKEN",
}

# ============================================================================
# TEXT GENERATION MODELS
# ============================================================================
TEXT_MODELS = {
    "cerebras": {
        "api_key":    "YOUR_CEREBRAS_API_KEY",
        "api_url":    "https://api.cerebras.ai/v1/chat/completions",
        "model_name": "gpt-oss-120b"
    },
    "groq": {
        "api_key":    "YOUR_GROQ_API_KEY",
        "api_url":    "https://api.groq.com/openai/v1/chat/completions",
        "model_name": "llama-3.3-70b-versatile"
    },
    "mistral": {
        "api_key":    "YOUR_MISTRAL_API_KEY",
        "api_url":    "https://api.mistral.ai/v1/chat/completions",
        "model_name": "mistral-large-latest"
    }
}

# ============================================================================
# IMAGE GENERATION MODELS
# ============================================================================
IMAGE_MODELS = {
    "black_forest_labs": {
        "api_key":    "YOUR_BFL_API_KEY",
        "api_url":    "https://api.bfl.ml/v1/image-generation",
        "model_name": "flux-pro"
    },
    "stability_ai": {
        "api_key":    "YOUR_STABILITY_AI_API_KEY",
        "api_url":    "https://api.stability.ai/v1/generation",
        "model_name": "stable-diffusion-xl-1024-v1-0"
    },
    "huggingface": {
        "api_key":    "YOUR_HUGGINGFACE_API_KEY",
        "api_url":    "https://api-inference.huggingface.co/models",
        "model_name": "runwayml/stable-diffusion-v1-5"
    },
    "pixazo": {
        "api_key":    "YOUR_PIXAZO_API_KEY",
        "api_url":    "https://api.pixazo.ai/v1/image-generation",
        "model_name": "flux"
    }
}

# ============================================================================
# AUDIO GENERATION MODELS (Text-to-Speech)
# ============================================================================
AUDIO_MODELS = {
    "gemini_tts": {
        "api_key":    "YOUR_GEMINI_API_KEY",
        "model_name": "gemini-2.5-flash-preview-tts"
    },
    "deepgram": {
        "api_key":    "YOUR_DEEPGRAM_API_KEY",
        "model_name": "aura-asteria-en"
    }
}

# ============================================================================
# BIG PLAYERS - BENCHMARK VALUES
# Sources:
#   TEXT  - BERTScore/ROUGE: arxiv.org/pdf/2506.20274 (Enterprise LLM Eval Benchmark 2025)
#           Readability: measured on same FK-grade/10 scale as small models
#   IMAGE - CLIP/Aesthetic: arxiv.org/pdf/2509.12750 (Human & MLLM Image Preference 2025)
#   AUDIO - MOS: blog.promptlayer.com GPT-4o-Mini-TTS report; Gemini UTMOS evals 2025
#           WER: EmergentTTS-Eval arxiv.org/pdf/2505.23009
# ============================================================================
BIG_PLAYERS_BENCHMARKS = {
    "claude": {
        "bertscore":   0.851,
        "rouge":       0.078,
        "readability": 1.85,
        "clip":        0.312,
        "aesthetic":   7.20,
        "clap":        0.74,
        "mos":         4.10,
        "wer":         0.072
    },
    "chatgpt": {
        "bertscore":   0.832,
        "rouge":       0.071,
        "readability": 1.72,
        "clip":        0.305,
        "aesthetic":   7.05,
        "clap":        0.72,
        "mos":         4.05,
        "wer":         0.085
    },
    "gemini": {
        "bertscore":   0.796,
        "rouge":       0.068,
        "readability": 1.68,
        "clip":        0.298,
        "aesthetic":   6.90,
        "clap":        0.76,
        "mos":         4.20,
        "wer":         0.068
    }
}

# ============================================================================
# OUTPUT FOLDER STRUCTURE
# ============================================================================
from pathlib import Path as _Path
_BASE = _Path(__file__).parent

OUTPUT_DIRS = {
    "text":   str(_BASE / "outputs/text"),
    "image":  str(_BASE / "outputs/images"),
    "audio":  str(_BASE / "outputs/audio"),
    "scores": str(_BASE / "outputs/scores"),
    "graphs": str(_BASE / "outputs/graphs"),
}

# ============================================================================
# SCORING PARAMETERS
# ============================================================================
SCORING_CONFIG = {
    "bert_score":  {"model": "microsoft/deberta-xlarge-mnli", "lang": "en"},
    "rouge":       {"rouge_types": ["rouge1", "rougeL"], "use_stemmer": True},
    "readability": {"metric": "flesch_kincaid"},
    "clip":        {"model": "openai/clip-vit-large-patch14"},
    "aesthetic":   {"model": "aesthetics"},
    "clap":        {"model": "laion/clap-htsat-unfused"},
    "mos":         {"model": "audiocaps"},
    "wer":         {"language": "en"}
}
