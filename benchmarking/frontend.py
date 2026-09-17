import sys
import os
import base64
import time
import importlib.util
from pathlib import Path

import streamlit as st
import requests

# ---------------------------
# Load credentials
#
# Loading via importlib with an alias avoids the stdlib `secrets` shadow bug
# that hit this project twice. The alias "project_secrets" means Python never
# registers it as the stdlib `secrets` module, so transformers/FastAPI are safe.
# ---------------------------
BASE_DIR = Path(os.path.abspath(__file__)).parent
_spec = importlib.util.spec_from_file_location("project_secrets", BASE_DIR.parent / "secrets.py")
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CF_ACCOUNT_ID = _mod.CLOUDFLARE_ACCOUNT_ID
CF_API_TOKEN  = _mod.CLOUDFLARE_API_TOKEN
GROQ_KEY      = getattr(_mod, "GROQ_KEY", None)
GEMINI_KEY    = getattr(_mod, "GEMINI_API_KEY", None)

CF_BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run"
CF_HEADERS  = {
    "Authorization": f"Bearer {CF_API_TOKEN}",
    "Content-Type": "application/json",
}


# ---------------------------
# Provider clients (created lazily so a missing key never breaks startup)
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_groq():
    if not GROQ_KEY:
        return None
    from openai import OpenAI
    return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_KEY)


@st.cache_resource(show_spinner=False)
def get_gemini():
    if not GEMINI_KEY:
        return None
    from google import genai
    return genai.Client(api_key=GEMINI_KEY)

# ---------------------------
# Model Options
# ---------------------------
TEXT_MODELS = {
    "Llama 3.1 8B (Cloudflare)":   {"provider": "cloudflare", "id": "@cf/meta/llama-3.1-8b-instruct"},
    "Llama 4 Scout (Cloudflare)":  {"provider": "cloudflare", "id": "@cf/meta/llama-4-scout-17b-16e-instruct"},
    "Kimi K2.6 (Cloudflare)":      {"provider": "cloudflare", "id": "@cf/moonshotai/kimi-k2.6"},
    "GPT-OSS 20B (Cloudflare)":    {"provider": "cloudflare", "id": "@cf/openai/gpt-oss-20b"},
    "Llama 3.3 70B (Groq)":        {"provider": "groq",       "id": "llama-3.3-70b-versatile"},
    "Llama 3.1 8B Instant (Groq)": {"provider": "groq",       "id": "llama-3.1-8b-instant"},
    "Gemini Flash (Google)":       {"provider": "gemini",     "id": "gemini-flash-latest"},
}

IMAGE_MODELS = {
    "Flux 1 Schnell (Cloudflare)":        {"provider": "cloudflare", "id": "@cf/black-forest-labs/flux-1-schnell"},
    "Flux 2 Dev (Cloudflare)":            {"provider": "cloudflare", "id": "@cf/black-forest-labs/flux-2-dev"},
    "Leonardo Lucid Origin (Cloudflare)": {"provider": "cloudflare", "id": "@cf/leonardo/lucid-origin"},
    "Leonardo Phoenix 1.0 (Cloudflare)":  {"provider": "cloudflare", "id": "@cf/leonardo/phoenix-1.0"},
    "Gemini Flash Image (Google)":        {"provider": "gemini",     "id": "gemini-2.5-flash-image"},
}

AUDIO_MODELS = {
    "MeloTTS (Cloudflare)":         {"provider": "cloudflare", "id": "@cf/myshell-ai/melotts"},
    "Deepgram Aura-1 (Cloudflare)": {"provider": "cloudflare", "id": "@cf/deepgram/aura-1"},
    "Deepgram Aura-2 (Cloudflare)": {"provider": "cloudflare", "id": "@cf/deepgram/aura-2-es"},
    "Gemini TTS (Google)":          {"provider": "gemini",     "id": "gemini-2.5-flash-preview-tts"},
}


# ---------------------------
# Generation Functions
# ---------------------------
TEXT_INSTRUCTION = "Describe this scene in vivid detail in 3-4 sentences: {p}"
TEXT_MAX_TOKENS  = 200


def trim_to_sentence(text):
    """Drop a trailing fragment left by the token cap ("...The lake,")."""
    import re
    text = (text or "").strip()
    m = list(re.finditer(r"[.!?][\"')\]]*(?=\s|$)", text))
    return text[: m[-1].end()].strip() if m else text


def tts_text(text):
    t = trim_to_sentence(text)
    return t if len(t) <= 900 else trim_to_sentence(t[:900]) or t[:900]


def generate_text(prompt, model_info):
    """Route text generation by provider. Returns (text, error)."""
    provider, model_id = model_info["provider"], model_info["id"]
    try:
        if provider == "cloudflare":
            r = requests.post(
                f"{CF_BASE_URL}/{model_id}",
                headers=CF_HEADERS,
                json={
                    "messages": [{"role": "user", "content": TEXT_INSTRUCTION.format(p=prompt)}],
                    "max_tokens": TEXT_MAX_TOKENS,
                },
                timeout=60,
            )
            if r.status_code == 200:
                return trim_to_sentence(r.json().get("result", {}).get("response", "")), None
            return None, f"Error {r.status_code}: {r.text[:200]}"

        if provider == "groq":
            client = get_groq()
            if client is None:
                return None, "GROQ_KEY missing from api_keys.py"
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": TEXT_INSTRUCTION.format(p=prompt)}],
                max_tokens=TEXT_MAX_TOKENS,
            )
            return trim_to_sentence(resp.choices[0].message.content), None

        if provider == "gemini":
            client = get_gemini()
            if client is None:
                return None, "GEMINI_API_KEY missing from api_keys.py"
            resp = client.models.generate_content(
                model=model_id,
                contents=TEXT_INSTRUCTION.format(p=prompt),
            )
            return trim_to_sentence(resp.text), None

        return None, f"Unknown provider: {provider}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def generate_image(prompt, model_info):
    """Route image generation by provider. Returns (png_bytes, error)."""
    provider, model_id = model_info["provider"], model_info["id"]
    try:
        if provider == "cloudflare":
            r = requests.post(
                f"{CF_BASE_URL}/{model_id}",
                headers=CF_HEADERS,
                json={"prompt": prompt},
                timeout=120,
            )
            if r.status_code == 200:
                result = r.json().get("result", {})
                img_b64 = result.get("image", "")
                if img_b64:
                    return base64.b64decode(img_b64), None
                if r.content:
                    return r.content, None
            return None, f"Error {r.status_code}: {r.text[:200]}"

        if provider == "gemini":
            client = get_gemini()
            if client is None:
                return None, "GEMINI_API_KEY missing from api_keys.py"
            resp = client.models.generate_content(model=model_id, contents=prompt)
            for part in resp.candidates[0].content.parts:
                inline = getattr(part, "inline_data", None)
                if inline and inline.data:
                    return inline.data, None
            return None, "Gemini returned no image data"

        return None, f"Unknown provider: {provider}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def generate_audio(text, model_info):
    """Route TTS by provider. Returns (audio_bytes, error)."""
    provider, model_id = model_info["provider"], model_info["id"]
    clean = tts_text(text)
    try:
        if provider == "cloudflare":
            payload = {"text": clean} if "deepgram" in model_id else {"prompt": clean}
            r = requests.post(
                f"{CF_BASE_URL}/{model_id}", headers=CF_HEADERS, json=payload, timeout=60
            )
            if r.status_code == 200:
                try:
                    result = r.json().get("result", {})
                    audio_b64 = result.get("audio", "")
                    if audio_b64:
                        return base64.b64decode(audio_b64), None
                except Exception:
                    pass
                if r.content:
                    return r.content, None
            return None, f"Error {r.status_code}: {r.text[:200]}"
        if provider == "gemini":
            client = get_gemini()
            if client is None:
                return None, "GEMINI_API_KEY missing from api_keys.py"
            from google.genai import types
            resp = client.models.generate_content(
                model=model_id,
                contents=clean,
                config=types.GenerateContentConfig(response_modalities=["AUDIO"]),
            )
            for part in resp.candidates[0].content.parts:
                inline = getattr(part, "inline_data", None)
                if inline and inline.data:
                    return _pcm_to_wav(inline.data), None
            return None, "Gemini returned no audio data"

        return None, f"Unknown provider: {provider}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _pcm_to_wav(pcm_bytes, sample_rate=24000, channels=1, bits=16):
    """Gemini TTS returns raw PCM; wrap it in a WAV header so st.audio can play it."""
    import struct
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    header = (
        b"RIFF" + struct.pack("<I", 36 + len(pcm_bytes)) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                                byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", len(pcm_bytes))
    )
    return header + pcm_bytes


# =========================================================================
# SCORING
# =========================================================================
JUDGE_SPLIT = 3.5
LAMBDA      = 0.5

LR_PARAMS = {
    "mu_good": [0.8400, 0.4656, 0.8723],
    "sd_good": [0.0088, 0.0255, 0.0153],
    "mu_bad":  [0.8372, 0.4585, 0.8706],
    "sd_bad":  [0.0097, 0.0245, 0.0153],
    "r": 0.1604, "pairwise": 57.4,
    "label": "quality features (Q_text, Q_image, Q_audio)",
}


def quality_pair(a, b, lam=LAMBDA):
    avg = (a + b) / 2.0
    var = ((a - avg) ** 2 + (b - avg) ** 2) / 2.0
    return float(min(max(avg - lam * var, 0.0), 1.0))


def likelihood_ratio(f):
    """S(p) = sum_i [ log P(f_i|Good) - log P(f_i|Not-Good) ]. >0 => likely Good."""
    import math
    p = LR_PARAMS
    mu_g, sd_g, mu_b, sd_b = p["mu_good"], p["sd_good"], p["mu_bad"], p["sd_bad"]
    s = 0.0
    for i in range(3):
        s += (((f[i] - mu_b[i]) ** 2) / (2 * sd_b[i] ** 2)
              - ((f[i] - mu_g[i]) ** 2) / (2 * sd_g[i] ** 2)
              + math.log(sd_b[i] / sd_g[i]))
    return float(s)


@st.cache_resource(show_spinner="Loading quality models (first run only)â€¦")
def _load_quality_models():
    from transformers import CLIPModel, CLIPProcessor, pipeline as hf_pipeline
    import whisper
    return {
        "clip":      CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval(),
        "clip_proc": CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
        "aesthetic": hf_pipeline("image-classification", model="cafeai/cafe_aesthetic"),
        "whisper":   whisper.load_model("base"),
    }


def compute_scores(prompt, text_out, image_bytes, audio_bytes, spoken_text):
    """Returns a dict of per-generation metrics, or {'error': ...}."""
    import io, torch, numpy as np
    from PIL import Image
    try:
        M = _load_quality_models()

        from bert_score import score as bert_score_fn

        def _bertscore_f1(candidate, reference):
            if not (candidate or "").strip():
                return 0.0
            _, _, f1 = bert_score_fn([candidate], [reference], lang="en", verbose=False)
            return float(f1.mean())

        q_text = _bertscore_f1(text_out, prompt)

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        ti = M["clip_proc"](text=[prompt], return_tensors="pt", padding=True,
                            truncation=True, max_length=77)
        ii = M["clip_proc"](images=img, return_tensors="pt")
        with torch.no_grad():
            clip_out = M["clip"](input_ids=ti["input_ids"],
                                 attention_mask=ti["attention_mask"],
                                 pixel_values=ii["pixel_values"])
        te = clip_out.text_embeds.squeeze().numpy()
        ie = clip_out.image_embeds.squeeze().numpy()
        clip_s = float(np.clip(np.dot(te, ie) / (np.linalg.norm(te) * np.linalg.norm(ie)), 0, 1))
        aes = next((r["score"] for r in M["aesthetic"](img) if r["label"] == "aesthetic"), 0.0)
        q_image = quality_pair(clip_s, float(aes))

        import soundfile as sf
        from scipy import signal as scipy_signal
        wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = scipy_signal.resample(wav, int(len(wav) * 16000 / sr)).astype("float32")
        transcript = M["whisper"].transcribe(np.ascontiguousarray(wav))["text"].strip()
        semantic = _bertscore_f1(transcript, prompt)
        from jiwer import wer as jiwer_wer
        wer_inv = 1.0 - float(np.clip(jiwer_wer(spoken_text.lower(), (transcript or "").lower()), 0, 1))
        q_audio = quality_pair(semantic, wer_inv)

        out = {"q_text": q_text, "q_image": q_image, "q_audio": q_audio,
               "clip": clip_s, "aesthetic": float(aes), "transcript": transcript}
        out["lr_score"] = likelihood_ratio([q_text, q_image, q_audio])
        return out
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def llm_judge(prompt, text_out, image_bytes, audio_bytes, transcript):
    """Ask Gemini to rate the triple 1-5, same rubric as the 500-prompt run."""
    client = get_gemini()
    if client is None:
        return None, "GEMINI_API_KEY missing from api_keys.py"
    import json, re
    rubric = (
        "Rate this AI-generated set (text, image, audio narration) for the prompt below.\n"
        "Score 1-5 where 5 = all three excellent and consistent with the prompt, "
        "1 = largely irrelevant.\n"
        'Respond ONLY with JSON: {"overall": N, "text": N, "image": N, "audio": N, '
        '"reasoning": "one sentence"}\n\n'
        f'Prompt: "{prompt}"\n\nGenerated text:\n{(text_out or "")[:1200]}\n\n'
        f'The narration audio is attached. It reads the generated text aloud.'
        + (f'\nWhisper transcript of it:\n{transcript[:400]}' if transcript else "")
    )
    import time as _time
    from google.genai import types
    parts = [rubric]
    if image_bytes:
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
    if audio_bytes:
        mime = "audio/wav" if audio_bytes[:4] == b"RIFF" else "audio/mp3"
        parts.append(types.Part.from_bytes(data=audio_bytes, mime_type=mime))
    cfg = types.GenerateContentConfig(temperature=0.1, max_output_tokens=2048)

    last_err = None
    for model_id in ("gemini-flash-latest", "gemini-3.6-flash", "gemini-flash-lite-latest"):
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=model_id, contents=parts, config=cfg)
                m = re.search(r"\{.*\}", resp.text or "", re.DOTALL)
                if not m:
                    last_err = f"{model_id} returned no JSON"
                    break
                return json.loads(m.group()), None
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                transient = any(c in str(e) for c in ("503", "UNAVAILABLE", "429",
                                                      "RESOURCE_EXHAUSTED", "500"))
                if not transient:
                    break
                if attempt < 2:
                    _time.sleep(2 ** attempt)
    return None, f"all judge models unavailable â€” last error: {last_err}"


# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="AI Content Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------
# Background artwork
# ---------------------------
@st.cache_data(show_spinner=False)
def _asset_uri(filename: str) -> str:
    path = Path(__file__).parent / "assets" / filename
    if not path.exists():
        return ""
    mime = {
        ".webp": "image/webp",
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(path.suffix.lower(), "image/png")
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


HERO_IMAGE    = "dashboard_bg.webp"
SIDEBAR_IMAGE = "dashboard_bg.webp"
EMPTY_IMAGE   = "empty_state.webp"

BG_URI    = _asset_uri(HERO_IMAGE)
SIDE_URI  = _asset_uri(SIDEBAR_IMAGE)
EMPTY_URI = _asset_uri(EMPTY_IMAGE)


# ---------------------------
# CSS â€” Obsidian theme
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
* { font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important; }

/* â”€â”€ Page ground â”€â”€ */
.stApp { background: #03070F !important; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2.2rem !important;
    padding-right: 2.2rem !important;
    max-width: 100% !important;
}
header[data-testid="stHeader"] { background: transparent !important; }

/* â”€â”€ Global text overrides (dark ground) â”€â”€ */
.stApp p, .stApp li, .stApp div, .stApp span { color: #CBD5E1; }
.stMarkdown p, .stMarkdown li { color: #94A3B8; }

/* â”€â”€ Metrics â”€â”€ */
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    color: #475569 !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #F1F5F9 !important;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * { color: #334155 !important; }
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * { color: #94A3B8 !important; }

/* â”€â”€ Sidebar â”€â”€ */
[data-testid="stSidebar"] {
    background: #020509 !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
[data-testid="stSidebar"] label {
    color: #334155 !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
}
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: #CBD5E1 !important;
}
[data-testid="stSidebar"] .stSelectbox span { color: #CBD5E1 !important; }
[data-testid="stSidebar"] [data-testid="stCheckbox"] p,
[data-testid="stSidebar"] [data-testid="stCheckbox"] span,
[data-testid="stSidebar"] .stMarkdown p { color: #64748B !important; }
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg,
[data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"] svg {
    fill: #334155 !important; color: #334155 !important;
}

/* â”€â”€ Hero â”€â”€ */
.hero {
    position: relative;
    border-radius: 20px;
    overflow: hidden;
    padding: 2.4rem 2.6rem 2.1rem;
    margin-bottom: 1.8rem;
    background: linear-gradient(135deg, #04091A 0%, #0C1C38 45%, #112444 100%);
    border: 1px solid rgba(255,255,255,0.07);
    box-shadow: 0 32px 80px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.07);
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 50%, #06B6D4 100%);
    border-radius: 20px 20px 0 0;
}
.hero-title {
    font-size: clamp(1.9rem, 3.2vw, 2.5rem);
    font-weight: 800;
    letter-spacing: -0.025em;
    line-height: 1.1;
    background: linear-gradient(135deg, #FFFFFF 0%, #93C5FD 55%, #C4B5FD 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
}
.hero-sub {
    font-size: 0.93rem;
    color: #4A5C7A;
    font-weight: 500;
    letter-spacing: 0.01em;
}
@media (max-width: 640px) {
    .hero { padding: 1.6rem 1.4rem; }
}

/* â”€â”€ Composer card â”€â”€ */
.composer-wrap {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.5rem 1.8rem 1.3rem;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.35);
    margin-bottom: 2rem;
}

/* â”€â”€ Prompt chip â”€â”€ */
.prompt-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(99,102,241,0.10);
    border: 1px solid rgba(99,102,241,0.22);
    border-radius: 99px;
    padding: 0.42rem 1.1rem;
    font-size: 0.86rem;
    color: #A5B4FC;
    font-weight: 600;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
    letter-spacing: 0.005em;
}

/* â”€â”€ Card labels â”€â”€ */
.card-label {
    font-size: 0.66rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.9rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.label-text  { color: #38BDF8; }
.label-image { color: #C084FC; }
.label-audio { color: #4ADE80; }

/* â”€â”€ Divider â”€â”€ */
.divider {
    height: 1px;
    background: rgba(255,255,255,0.04);
    margin: 2rem 0 2.2rem;
}

/* â”€â”€ Empty state â”€â”€ */
.empty-state {
    text-align: center;
    padding: 3rem 1rem 4rem;
}
.empty-icon { font-size: 2.2rem; margin-bottom: 0.6rem; opacity: 0.35; }
.empty-art {
    display: block;
    width: 100%;
    max-width: 500px;
    margin: 0 auto 1.8rem;
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    opacity: 0.80;
}
.empty-text { font-size: 1rem; font-weight: 600; color: #1E293B; }

/* â”€â”€ Text area â”€â”€ */
.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 14px !important;
    color: #E2E8F0 !important;
    font-size: 0.94rem !important;
    resize: none !important;
    caret-color: #60A5FA;
}
.stTextArea textarea:focus {
    border-color: rgba(96,165,250,0.4) !important;
    box-shadow: 0 0 0 3px rgba(96,165,250,0.10) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder { color: #1E293B !important; }

/* â”€â”€ Generate button â”€â”€ */
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #1D4ED8 0%, #4F46E5 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 2.2rem !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 22px rgba(79,70,229,0.38) !important;
    transition: box-shadow 0.2s ease, transform 0.15s ease !important;
}
.stFormSubmitButton > button:hover {
    box-shadow: 0 6px 32px rgba(79,70,229,0.55) !important;
    transform: translateY(-1px) !important;
}

/* â”€â”€ Images â”€â”€ */
[data-testid="stImage"] img {
    border-radius: 12px;
    box-shadow: 0 8px 28px rgba(0,0,0,0.4);
}

/* â”€â”€ Spinner â”€â”€ */
.stSpinner > div { border-top-color: #3B82F6 !important; }

/* â”€â”€ Scrollbar â”€â”€ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.10); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.18); }

/* â”€â”€ Alert / warning â”€â”€ */
.stAlert { border-radius: 12px !important; }

</style>
""", unsafe_allow_html=True)

if BG_URI:
    st.markdown("""
<style>
.stApp {
    background-image:
        linear-gradient(rgba(3,7,15,0.97), rgba(3,7,15,0.97)),
        url("__BG__");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
}
.hero {
    background-image:
        linear-gradient(135deg,
            rgba(4,9,26,0.98) 0%,
            rgba(9,20,44,0.92) 40%,
            rgba(9,20,44,0.55) 75%,
            rgba(9,20,44,0.22) 100%),
        url("__BG__");
    background-size: cover, cover;
    background-position: center, center right;
}
[data-testid="stSidebar"] {
    background-image:
        linear-gradient(180deg,
            rgba(2,5,9,0.97) 0%,
            rgba(2,5,9,0.99) 65%,
            rgba(2,5,9,1.00) 100%),
        url("__SIDE__") !important;
    background-size: cover !important;
    background-position: 32% center !important;
    background-repeat: no-repeat !important;
    background-attachment: local !important;
}
</style>
""".replace("__BG__", BG_URI).replace("__SIDE__", SIDE_URI or BG_URI), unsafe_allow_html=True)


# ---------------------------
# Sidebar â€” Model Selection
# ---------------------------
with st.sidebar:
    st.markdown(
        "<div style='padding:1.4rem 0 0.6rem; font-size:0.95rem; font-weight:800;"
        " color:#E2E8F0; letter-spacing:-0.01em;'>âš™ Models</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='height:1px; background:rgba(255,255,255,0.06); margin-bottom:1rem;'></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size:0.65rem; font-weight:700; color:#334155;"
        " text-transform:uppercase; letter-spacing:0.10em; margin-bottom:0.35rem;'>"
        "Text</div>",
        unsafe_allow_html=True,
    )
    selected_text_model  = st.selectbox("Text model",  list(TEXT_MODELS.keys()),  label_visibility="collapsed")

    st.markdown(
        "<div style='height:1px; background:rgba(255,255,255,0.05); margin:0.8rem 0;'></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.65rem; font-weight:700; color:#334155;"
        " text-transform:uppercase; letter-spacing:0.10em; margin-bottom:0.35rem;'>"
        "Image</div>",
        unsafe_allow_html=True,
    )
    selected_image_model = st.selectbox("Image model", list(IMAGE_MODELS.keys()), label_visibility="collapsed")

    st.markdown(
        "<div style='height:1px; background:rgba(255,255,255,0.05); margin:0.8rem 0;'></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.65rem; font-weight:700; color:#334155;"
        " text-transform:uppercase; letter-spacing:0.10em; margin-bottom:0.35rem;'>"
        "Audio</div>",
        unsafe_allow_html=True,
    )
    selected_audio_model = st.selectbox("Audio model", list(AUDIO_MODELS.keys()), label_visibility="collapsed")

    st.markdown(
        "<div style='height:1px; background:rgba(255,255,255,0.05); margin:1rem 0 0.6rem;'></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-size:0.7rem; color:#1E293B; line-height:1.7;"
        f" font-family:JetBrains Mono, monospace;'>"
        f"<span style='color:#1D3A5C;'>{TEXT_MODELS[selected_text_model]['id']}</span><br>"
        f"<span style='color:#1D3A5C;'>{IMAGE_MODELS[selected_image_model]['id']}</span><br>"
        f"<span style='color:#1D3A5C;'>{AUDIO_MODELS[selected_audio_model]['id']}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='height:1px; background:rgba(255,255,255,0.05); margin:1.2rem 0 0.8rem;'></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.65rem; font-weight:700; color:#334155;"
        " text-transform:uppercase; letter-spacing:0.10em; margin-bottom:0.5rem;'>"
        "Scoring</div>",
        unsafe_allow_html=True,
    )

    do_quality = st.checkbox("Quality + likelihood ratio", value=False,
                             help="Q_text / Q_image / Q_audio plus the likelihood-ratio "
                                  "score. Loads ~3 GB on first use. r = 0.1604.")
    do_judge = st.checkbox("LLM judge (Gemini)", value=False,
                           help="Sends the generated triple to Gemini for a 1â€“5 rating. "
                                "No local models needed.")


# ---------------------------
# Session State
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------
# Page Header
# ---------------------------
st.markdown(
    '<div class="hero">'
    '<div class="hero-title">AI Content Generator</div>'
    '<div class="hero-sub">One prompt &rarr; text &middot; image &middot; audio</div>'
    '</div>',
    unsafe_allow_html=True,
)


# ---------------------------
# Prompt Composer
# ---------------------------
st.markdown('<div class="composer-wrap">', unsafe_allow_html=True)
with st.form("prompt_form", clear_on_submit=True):
    user_input = st.text_area(
        "prompt",
        height=90,
        placeholder="Describe a scene, topic, or idea â€” e.g. A mountain lake surrounded by pine trees at sunrise",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("âœ¦  Generate text Â· image Â· audio", type="primary")
st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------
# Generation
# ---------------------------
if submitted and user_input.strip():
    text_out = image_out = audio_out = None
    text_err = image_err = audio_err = None
    text_time = image_time = audio_time = 0.0

    text_model_id  = TEXT_MODELS[selected_text_model]
    image_model_id = IMAGE_MODELS[selected_image_model]
    audio_model_id = AUDIO_MODELS[selected_audio_model]

    c1, c2, c3 = st.columns(3)

    with c1:
        with st.spinner(f"Writing text with {selected_text_model}..."):
            t0 = time.perf_counter()
            text_out, text_err = generate_text(user_input, text_model_id)
            text_time = time.perf_counter() - t0

    with c2:
        with st.spinner(f"Creating image with {selected_image_model}..."):
            t1 = time.perf_counter()
            image_out, image_err = generate_image(user_input, image_model_id)
            image_time = time.perf_counter() - t1

    with c3:
        with st.spinner(f"Generating audio with {selected_audio_model}..."):
            t2 = time.perf_counter()
            audio_out, audio_err = generate_audio(text_out or user_input, audio_model_id)
            audio_time = time.perf_counter() - t2

    scores = None
    judge = None
    have_all = bool(text_out and image_out and audio_out)

    if (do_quality or do_judge) and not have_all:
        st.warning("Scoring skipped â€” it needs all three outputs to succeed.")
    elif do_quality and have_all:
        with st.spinner("Scoring outputsâ€¦"):
            t3 = time.perf_counter()
            scores = compute_scores(user_input, text_out, image_out, audio_out,
                                    spoken_text=tts_text(text_out or user_input))
            if scores:
                scores["score_time"] = time.perf_counter() - t3

    if do_judge and have_all:
        with st.spinner("Asking the judgeâ€¦"):
            judge, judge_err = llm_judge(
                user_input, text_out, image_out, audio_out,
                (scores or {}).get("transcript", ""),
            )
            if judge_err:
                judge = {"error": judge_err}

    st.session_state.history.append({
        "prompt":      user_input.strip(),
        "text":        text_out,
        "text_err":    text_err,
        "text_time":   text_time,
        "text_model":  selected_text_model,
        "image":       image_out,
        "image_err":   image_err,
        "image_time":  image_time,
        "image_model": selected_image_model,
        "audio":       audio_out,
        "audio_err":   audio_err,
        "audio_time":  audio_time,
        "audio_model": selected_audio_model,
        "scores":      scores,
        "judge":       judge,
    })
    st.rerun()


# ---------------------------
# History
# ---------------------------
if not st.session_state.history:
    art = (f'<img class="empty-art" src="{EMPTY_URI}" alt="">'
           if EMPTY_URI else '<div class="empty-icon">âœ¦</div>')
    st.markdown(f"""
    <div class="empty-state">
        {art}
        <div class="empty-text">Enter a prompt to generate text, image and audio together</div>
    </div>
    """, unsafe_allow_html=True)

for entry in reversed(st.session_state.history):
    st.markdown(f"<div class='prompt-chip'>&#x2764; {entry['prompt']}</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f'<div class="card-label label-text">&#x25C6; Text &mdash; {entry.get("text_model","")}</div>', unsafe_allow_html=True)
        if entry.get("text_err"):
            st.error(entry["text_err"])
        elif entry.get("text"):
            st.markdown(
                f"<div style='color:#94A3B8; font-size:0.93rem; line-height:1.7;'>{entry['text']}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Generated in {entry['text_time']:.1f}s")

    with c2:
        st.markdown(f'<div class="card-label label-image">&#x25C6; Image &mdash; {entry.get("image_model","")}</div>', unsafe_allow_html=True)
        if entry.get("image_err"):
            st.error(entry["image_err"])
        elif entry.get("image"):
            st.image(entry["image"], use_column_width=True)
            st.caption(f"Generated in {entry['image_time']:.1f}s")

    with c3:
        st.markdown(f'<div class="card-label label-audio">&#x25C6; Audio &mdash; {entry.get("audio_model","")}</div>', unsafe_allow_html=True)
        if entry.get("audio_err"):
            st.error(entry["audio_err"])
        elif entry.get("audio"):
            st.audio(entry["audio"], format="audio/wav")
            st.caption(f"Generated in {entry['audio_time']:.1f}s")

    sc = entry.get("scores")
    jd = entry.get("judge")
    if sc or jd:
        st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

        if sc and sc.get("error"):
            st.warning(f"Scoring failed â€” {sc['error']}")
        elif sc:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Q_text",  f"{sc['q_text']:.4f}", help="BERTScore(generated text, prompt)")
            m2.metric("Q_image", f"{sc['q_image']:.4f}",
                      help=f"quality_pair(CLIP {sc['clip']:.3f}, aesthetic {sc['aesthetic']:.3f})")
            m3.metric("Q_audio", f"{sc['q_audio']:.4f}",
                      help="quality_pair(semantic, 1 âˆ’ WER) on the Whisper transcript")
            if "lr_score" in sc:
                verdict = "likely Good" if sc["lr_score"] > 0 else "likely Not-Good"
                m4.metric("Likelihood ratio", f"{sc['lr_score']:+.3f}", delta=verdict,
                          delta_color="normal" if sc["lr_score"] > 0 else "inverse",
                          help=f"S(p) = Î£ [log P(fáµ¢|Good) âˆ’ log P(fáµ¢|Not-Good)], "
                               f"computed on {LR_PARAMS['label']}. >0 predicts Good.")
            else:
                m4.metric("Likelihood ratio", "â€”", help="Enable 'Quality scores'")

        if jd:
            if jd.get("error"):
                st.warning(f"Judge unavailable â€” {jd['error']}")
            else:
                j1, j2, j3, j4 = st.columns(4)
                j1.metric("Judge â€” overall", f"{jd.get('overall','?')}/5")
                j2.metric("Judge â€” text",    f"{jd.get('text','?')}/5")
                j3.metric("Judge â€” image",   f"{jd.get('image','?')}/5")
                j4.metric("Judge â€” audio",   f"{jd.get('audio','?')}/5")
                if jd.get("reasoning"):
                    st.caption(f"Judge: {jd['reasoning']}")

        if sc and "lr_score" in sc:
            st.caption(
                f"Likelihood ratio: r = {LR_PARAMS['r']:.2f} against the judge over 500 prompts "
                f"({LR_PARAMS['pairwise']:.0f}% pairwise, 50% = chance) â€” indicative, not authoritative."
            )
        if sc and sc.get("score_time"):
            st.caption(f"Scored in {sc['score_time']:.1f}s")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)



