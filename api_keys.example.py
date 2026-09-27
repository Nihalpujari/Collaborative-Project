"""
Template for api_keys.py — the credentials file read by run_trio_local.py and hf_app.py.

    copy api_keys.example.py api_keys.py      (Windows)
    cp   api_keys.example.py api_keys.py      (macOS/Linux)

api_keys.py is listed in .gitignore; never commit it.
Do NOT name this file secrets.py: that shadows Python's built-in `secrets`
module and breaks transformers / huggingface_hub on import.
"""

# ── Cloudflare Workers AI (required) ──────────────────────────────────────────
# Dashboard → Workers AI → "Use REST API" gives the account ID and a token.
CLOUDFLARE_ACCOUNT_ID = "YOUR_CLOUDFLARE_ACCOUNT_ID"
CLOUDFLARE_API_TOKEN  = "YOUR_CLOUDFLARE_API_TOKEN"

# Optional: several free accounts (10,000 neurons/day each). The app tries them
# in order and moves on when one fails or runs out of quota.
CLOUDFLARE_POOL = [
    {"label": "account-1", "account_id": "YOUR_CLOUDFLARE_ACCOUNT_ID", "api_token": "YOUR_CLOUDFLARE_API_TOKEN"},
    # {"label": "account-2", "account_id": "...", "api_token": "..."},
]

# ── Optional providers (only needed if you pick these models in the app) ─────
GROQ_KEY       = ""   # https://console.groq.com/keys
GEMINI_API_KEY = ""   # https://aistudio.google.com/apikey  (Gemini text/TTS + LLM judge)
CEREBRAS_KEY   = ""
MISTRAL_KEY    = ""
