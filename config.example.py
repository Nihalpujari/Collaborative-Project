"""
Template for config.py — copy this to config.py and fill in your API keys.
    cp config.example.py config.py
config.py is gitignored and will never be committed.
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
        "api_key":    "YOUR_STABILITY_API_KEY",
        "api_url":    "https://api.stability.ai/v1/generation",
        "model_name": "stable-diffusion-xl-1024-v1-0"
    },
    "huggingface": {
        "api_key":    "YOUR_HF_API_KEY",
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
# AUDIO GENERATION MODELS
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
