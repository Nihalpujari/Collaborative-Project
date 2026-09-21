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
# NOTE: the file is api_keys.py, NOT secrets.py. A module named secrets.py
# shadows Python's standard-library `secrets`, which silently breaks
# huggingface_hub -> transformers -> the whole scoring stack. This project
# has hit that bug twice; do not rename it back.
# ---------------------------
BASE_DIR = Path(os.path.abspath(__file__)).parent
_spec = importlib.util.spec_from_file_location("project_api_keys", BASE_DIR.parent / "api_keys.py")
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
# Same instruction and token budget as pipeline/step1_generate.py. The
# likelihood-ratio Gaussians were fitted on text produced this way; a longer
# "write a story" prompt drifts Q_text ~3 sd below the training mean and
# gets cut off mid-sentence by the token cap.
TEXT_INSTRUCTION = "Describe this scene in vivid detail in 3-4 sentences: {p}"
TEXT_MAX_TOKENS  = 200


def trim_to_sentence(text):
    """Drop a trailing fragment left by the token cap ("...The lake,")."""
    import re
    text = (text or "").strip()
    m = list(re.finditer(r"[.!?][\"')\]]*(?=\s|$)", text))
    return text[: m[-1].end()].strip() if m else text


def tts_text(text):
    """The exact string handed to TTS. Sentence-aware cap so the audio never
    stops mid-word, and one place to compute it so scoring can compare the
    transcript against what was actually spoken."""
    t = trim_to_sentence(text)
    # 3-4 sentences at 200 tokens is ~600-700 chars; 900 covers it with room.
    # Deepgram Aura caps at 2000, MeloTTS and Gemini are well above that.
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
                result = r.json().get("result", {})
                audio_b64 = result.get("audio", "")
                if audio_b64:
                    return base64.b64decode(audio_b64), None
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
#
# Constants below ARE the fitted model. They come from the 500-prompt run
# (results/all_500_v2_summary.csv and the Gaussians fitted in
# pipeline/run_v2.py). Nothing is trained at runtime - only the features
# are computed. Do not hand-edit these without re-running the pipeline.
# =========================================================================
# W1, W2, W3 = 7.8666, 2.2790, 0.7796  - learned modality weights from the
# 500-prompt run. Used only to weight the ImageBind coherence features, which
# the app no longer computes. Kept here as a record of the fitted values.
JUDGE_SPLIT = 3.5                              # Good if judge rating > 3.5
LAMBDA      = 0.5

# Fitted likelihood-ratio parameters (V1), validated on the 500 prompts with
# leave-one-out CV. Features = [Q_text, Q_image, Q_audio] - no extra models.
#
# The study also fitted a V2 variant on quality-weighted ImageBind coherence
# features [s1_w, s2_w, s3_w]: r = 0.2046 vs. 0.1604 here. That looks better
# until you check ranking accuracy - 57.5% vs. 57.4%, a 0.1pp gain for a 4.5 GB
# download. The app therefore ships V1 only; V2 stays in the benchmark table
# below as a published result. See the report for the full comparison.
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
    """S(p) = sum_i [ log P(f_i|Good) - log P(f_i|Not-Good) ]. >0 => likely Good.

    f = [Q_text, Q_image, Q_audio]
    """
    import math
    p = LR_PARAMS
    mu_g, sd_g, mu_b, sd_b = p["mu_good"], p["sd_good"], p["mu_bad"], p["sd_bad"]
    s = 0.0
    for i in range(3):
        s += (((f[i] - mu_b[i]) ** 2) / (2 * sd_b[i] ** 2)
              - ((f[i] - mu_g[i]) ** 2) / (2 * sd_g[i] ** 2)
              + math.log(sd_b[i] / sd_g[i]))
    return float(s)


@st.cache_resource(show_spinner="Loading quality models (first run only)...")
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

        # ---- Q_text : BERTScore(generated text, prompt)
        from bert_score import score as bert_score_fn

        def _bertscore_f1(candidate, reference):
            """BERTScore F1, guarding the empty-candidate case.

            bert_score's sent_encode() falls back to the tokenizer's
            build_inputs_with_special_tokens() for empty strings, which
            transformers 5.x removed. An empty candidate has no semantic
            overlap anyway, so score it 0 instead of encoding bare
            special tokens.
            """
            if not (candidate or "").strip():
                return 0.0
            _, _, f1 = bert_score_fn([candidate], [reference], lang="en", verbose=False)
            return float(f1.mean())

        q_text = _bertscore_f1(text_out, prompt)

        # ---- Q_image : quality_pair(CLIP, aesthetic)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        ti = M["clip_proc"](text=[prompt], return_tensors="pt", padding=True,
                            truncation=True, max_length=77)
        ii = M["clip_proc"](images=img, return_tensors="pt")
        # transformers 5.x changed get_text_features/get_image_features to return
        # a BaseModelOutputWithPooling whose pooler_output is NOT projected into
        # CLIP's shared space. Take text_embeds/image_embeds off the full forward
        # instead - those are the projected embeddings the 500-prompt run used.
        with torch.no_grad():
            clip_out = M["clip"](input_ids=ti["input_ids"],
                                 attention_mask=ti["attention_mask"],
                                 pixel_values=ii["pixel_values"])
        te = clip_out.text_embeds.squeeze().numpy()
        ie = clip_out.image_embeds.squeeze().numpy()
        clip_s = float(np.clip(np.dot(te, ie) / (np.linalg.norm(te) * np.linalg.norm(ie)), 0, 1))
        aes = next((r["score"] for r in M["aesthetic"](img) if r["label"] == "aesthetic"), 0.0)
        q_image = quality_pair(clip_s, float(aes))

        # ---- Q_audio : quality_pair(semantic, 1 - WER)
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

        # Likelihood ratio on the quality features - free once the Q scores exist.
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
        # Attach the audio itself so the judge rates what it hears, not a
        # Whisper transcript (which is empty whenever scoring is off).
        mime = "audio/wav" if audio_bytes[:4] == b"RIFF" else "audio/mp3"
        parts.append(types.Part.from_bytes(data=audio_bytes, mime_type=mime))
    cfg = types.GenerateContentConfig(temperature=0.1, max_output_tokens=2048)

    # Flash is shared capacity and returns 503 under load. Retry with backoff,
    # then fall back to the other models this key can reach. "-latest" first so
    # we track Google's current Flash instead of pinning a version that 404s.
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
                    break                       # 404/400 -> next model, no retry
                if attempt < 2:
                    _time.sleep(2 ** attempt)   # 1s, 2s
    return None, f"all judge models unavailable - last error: {last_err}"


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
    """Return an image from assets/ as a data URI.

    Streamlit will not serve a local file to a CSS url(), so images have to
    be inlined. Cached so the base64 encode happens once per session.
    Returns "" if the file is missing, and the UI falls back to a plain
    gradient or hides the image rather than breaking.
    """
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


# Artwork used around the dashboard. Drop another file into assets/ and
# change the filename here to restyle - nothing else needs touching.
HERO_IMAGE    = "dashboard_bg.webp"   # alternative: "hero_alt.webp"
SIDEBAR_IMAGE = "dashboard_bg.webp"
EMPTY_IMAGE   = "empty_state.webp"

BG_URI    = _asset_uri(HERO_IMAGE)
SIDE_URI  = _asset_uri(SIDEBAR_IMAGE)
EMPTY_URI = _asset_uri(EMPTY_IMAGE)

# ---------------------------
# CSS
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: #F0F4F8; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}
header[data-testid="stHeader"] { background: transparent !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #1E293B !important; }
[data-testid="stSidebar"] label {
    color: #94A3B8 !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    background: #273549 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] .stSelectbox span { color: #E2E8F0 !important; }
[data-testid="stSidebar"] [data-testid="stCheckbox"] p,
[data-testid="stSidebar"] [data-testid="stCheckbox"] span,
[data-testid="stSidebar"] .stMarkdown p { color: #E2E8F0 !important; }
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg,
[data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"] svg { fill: #94A3B8 !important; color: #94A3B8 !important; }

.page-title { font-size: 2rem; font-weight: 800; color: #0F172A; }

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * { color: #64748B !important; }
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * { color: #0F172A !important; font-weight: 700; }
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * { color: #475569 !important; }
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * { color: #0F172A !important; }
.stMarkdown p, .stMarkdown li { color: #1E293B; }

.hero {
    position: relative;
    border-radius: 18px;
    overflow: hidden;
    padding: 2.1rem 2.2rem;
    margin-bottom: 1.4rem;
    background: linear-gradient(100deg, #0B1526 0%, #16294A 60%, #1E3A6B 100%);
    box-shadow: 0 8px 28px rgba(15,23,42,0.18);
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.01em;
    line-height: 1.15;
}
.hero-sub {
    font-size: 0.95rem;
    color: #C7D2E4;
    margin-top: 0.35rem;
}
@media (max-width: 640px) {
    .hero { padding: 1.5rem 1.2rem; }
    .hero-title { font-size: 1.5rem; }
}
.page-subtitle { font-size: 0.95rem; color: #64748B; margin-top: 0.3rem; margin-bottom: 1.5rem; }

.composer-wrap {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    padding: 1.4rem 1.6rem 1.1rem 1.6rem;
    box-shadow: 0 4px 20px rgba(15,23,42,0.07);
    margin-bottom: 2rem;
}
.prompt-chip {
    background: #EEF4FF;
    border: 1px solid #DBEAFE;
    border-radius: 10px;
    padding: 0.55rem 1rem;
    font-size: 0.88rem;
    color: #1E40AF;
    font-weight: 500;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
}
.card-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #F1F5F9;
}
.label-text  { color: #2563EB; }
.label-image { color: #7C3AED; }
.label-audio { color: #059669; }
.divider {
    height: 1px;
    background: linear-gradient(to right, #CBD5E1, transparent);
    margin: 1.5rem 0 2rem 0;
}
.empty-state { text-align: center; padding: 2.2rem 1rem 3rem; color: #94A3B8; }
.empty-icon  { font-size: 2.8rem; margin-bottom: 0.7rem; }
.empty-art {
    display: block;
    width: 100%;
    max-width: 540px;
    margin: 0 auto 1.5rem;
    border-radius: 16px;
    box-shadow: 0 12px 34px rgba(15,23,42,0.16);
}
.empty-text  { font-size: 1rem; font-weight: 500; }
.stFormSubmitButton > button {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.55rem 2rem !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.25) !important;
}
.stTextArea textarea {
    border-radius: 12px !important;
    border: 1px solid #E2E8F0 !important;
    font-size: 0.95rem !important;
    resize: none !important;
}
[data-testid="stImage"] img { border-radius: 10px; }
/* ARTWORK-INJECTION-POINT */

</style>
""", unsafe_allow_html=True)

# Layer the artwork in: full strength behind the hero, and a very faint
# fixed wash behind the whole page. The wash sits under a near-opaque
# tint so the dark image never fights the light UI underneath it.
if BG_URI:
    st.markdown("""
<style>
.stApp {
    background-image:
        linear-gradient(rgba(240,244,248,0.93), rgba(240,244,248,0.93)),
        url("__BG__");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
}
.hero {
    background-image:
        linear-gradient(100deg,
            rgba(9,17,33,0.95) 0%,
            rgba(9,17,33,0.82) 40%,
            rgba(9,17,33,0.35) 72%,
            rgba(9,17,33,0.20) 100%),
        url("__BG__");
    background-size: cover, cover;
    background-position: center, center right;
}

/* Sidebar - same artwork cropped to the bulb, fading to solid at the
   bottom so the dropdowns keep a clean surface to sit on.
   !important is needed to beat the flat colour set further up. */
[data-testid="stSidebar"] {
    background-image:
        linear-gradient(180deg,
            rgba(15,23,42,0.90) 0%,
            rgba(15,23,42,0.96) 60%,
            rgba(15,23,42,0.99) 100%),
        url("__SIDE__") !important;
    background-size: cover !important;
    background-position: 32% center !important;
    background-repeat: no-repeat !important;
    background-attachment: local !important;
}
</style>
""".replace("__BG__", BG_URI).replace("__SIDE__", SIDE_URI or BG_URI), unsafe_allow_html=True)


# ---------------------------
# Sidebar - Model Selection
# ---------------------------
with st.sidebar:
    st.markdown(
        "<div style='padding:1.2rem 0 0.5rem; font-size:1rem; font-weight:800; color:#F1F5F9;'>⚙️ Model Selection</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:1px; background:#334155; margin-bottom:1rem;'></div>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.7rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;'>Text Model</div>", unsafe_allow_html=True)
    selected_text_model  = st.selectbox("Text model",  list(TEXT_MODELS.keys()),  label_visibility="collapsed")

    st.markdown("<div style='height:1px; background:#334155; margin:0.8rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;'>Image Model</div>", unsafe_allow_html=True)
    selected_image_model = st.selectbox("Image model", list(IMAGE_MODELS.keys()), label_visibility="collapsed")

    st.markdown("<div style='height:1px; background:#334155; margin:0.8rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;'>Audio Model</div>", unsafe_allow_html=True)
    selected_audio_model = st.selectbox("Audio model", list(AUDIO_MODELS.keys()), label_visibility="collapsed")

    st.markdown("<div style='height:1px; background:#334155; margin:1rem 0 0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:0.75rem; color:#475569; line-height:1.6;'>"
        f"<b style='color:#94A3B8;'>Text</b><br>{TEXT_MODELS[selected_text_model]['id']}<br><br>"
        f"<b style='color:#94A3B8;'>Image</b><br>{IMAGE_MODELS[selected_image_model]['id']}<br><br>"
        f"<b style='color:#94A3B8;'>Audio</b><br>{AUDIO_MODELS[selected_audio_model]['id']}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ---- Scoring controls
    st.markdown("<div style='height:1px; background:#334155; margin:1.2rem 0 0.8rem;'></div>",
                unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem; font-weight:700; color:#64748B; "
                "text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;'>"
                "Scoring</div>", unsafe_allow_html=True)

    do_quality = st.checkbox("Quality + likelihood ratio", value=False,
                             help="Q_text / Q_image / Q_audio plus the likelihood-ratio "
                                  "score. Loads ~3 GB on first use. r = 0.1604.")
    do_judge = st.checkbox("LLM judge (Gemini)", value=False,
                           help="Sends the generated triple to Gemini for a 1-5 rating. "
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
    """
<div class="hero">
    <div class="hero-title">AI Content Generator</div>
    <div class="hero-sub">One prompt &rarr; text &middot; image &middot; audio</div>
</div>
""",
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
        placeholder="Describe a scene, topic, or idea - e.g. A mountain lake surrounded by pine trees at sunrise",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("✦  Generate text · image · audio", type="primary")
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

    # ---- Scoring (only when requested, and only if all three outputs exist)
    scores = None
    judge = None
    have_all = bool(text_out and image_out and audio_out)

    if (do_quality or do_judge) and not have_all:
        st.warning("Scoring skipped - it needs all three outputs to succeed.")
    elif do_quality and have_all:
        with st.spinner("Scoring outputs..."):
            t3 = time.perf_counter()
            scores = compute_scores(user_input, text_out, image_out, audio_out,
                                    spoken_text=tts_text(text_out or user_input))
            if scores:
                scores["score_time"] = time.perf_counter() - t3

    if do_judge and have_all:
        with st.spinner("Asking the judge..."):
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
           if EMPTY_URI else '<div class="empty-icon">✦</div>')
    st.markdown(f"""
    <div class="empty-state">
        {art}
        <div class="empty-text">Enter a prompt to generate text, image and audio together</div>
    </div>
    """, unsafe_allow_html=True)

for entry in reversed(st.session_state.history):
    st.markdown(f"<div class='prompt-chip'>💬 {entry['prompt']}</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f'<div class="card-label label-text">✦ Text — {entry.get("text_model","")}</div>', unsafe_allow_html=True)
        if entry.get("text_err"):
            st.error(entry["text_err"])
        elif entry.get("text"):
            st.markdown(
                f"<div style='color:#1E293B; font-size:0.93rem; line-height:1.6;'>{entry['text']}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Generated in {entry['text_time']:.1f}s")

    with c2:
        st.markdown(f'<div class="card-label label-image">✦ Image — {entry.get("image_model","")}</div>', unsafe_allow_html=True)
        if entry.get("image_err"):
            st.error(entry["image_err"])
        elif entry.get("image"):
            st.image(entry["image"], use_container_width=True)
            st.caption(f"Generated in {entry['image_time']:.1f}s")

    with c3:
        st.markdown(f'<div class="card-label label-audio">✦ Audio — {entry.get("audio_model","")}</div>', unsafe_allow_html=True)
        if entry.get("audio_err"):
            st.error(entry["audio_err"])
        elif entry.get("audio"):
            st.audio(entry["audio"], format="audio/wav")
            st.caption(f"Generated in {entry['audio_time']:.1f}s")

    # ---- Per-generation metrics
    sc = entry.get("scores")
    jd = entry.get("judge")
    if sc or jd:
        st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

        if sc and sc.get("error"):
            st.warning(f"Scoring failed - {sc['error']}")
        elif sc:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Q_text",  f"{sc['q_text']:.4f}", help="BERTScore(generated text, prompt)")
            m2.metric("Q_image", f"{sc['q_image']:.4f}",
                      help=f"quality_pair(CLIP {sc['clip']:.3f}, aesthetic {sc['aesthetic']:.3f})")
            m3.metric("Q_audio", f"{sc['q_audio']:.4f}",
                      help="quality_pair(semantic, 1 - WER) on the Whisper transcript")
            if "lr_score" in sc:
                verdict = "likely Good" if sc["lr_score"] > 0 else "likely Not-Good"
                m4.metric("Likelihood ratio", f"{sc['lr_score']:+.3f}", delta=verdict,
                          delta_color="normal" if sc["lr_score"] > 0 else "inverse",
                          help=f"S(p) = sum [log P(fi|Good) - log P(fi|Not-Good)], "
                               f"computed on {LR_PARAMS['label']}. >0 predicts Good.")
            else:
                m4.metric("Likelihood ratio", "—", help="Enable 'Quality scores'")

        if jd:
            if jd.get("error"):
                st.warning(f"Judge unavailable - {jd['error']}")
            else:
                j1, j2, j3, j4 = st.columns(4)
                j1.metric("Judge — overall", f"{jd.get('overall','?')}/5")
                j2.metric("Judge — text",    f"{jd.get('text','?')}/5")
                j3.metric("Judge — image",   f"{jd.get('image','?')}/5")
                j4.metric("Judge — audio",   f"{jd.get('audio','?')}/5")
                if jd.get("reasoning"):
                    st.caption(f"Judge: {jd['reasoning']}")

        # Reliability caveat - r is a corpus statistic, not a per-prompt value
        if sc and "lr_score" in sc:
            st.caption(
                f"Likelihood ratio: r = {LR_PARAMS['r']:.2f} against the judge over 500 prompts "
                f"({LR_PARAMS['pairwise']:.0f}% pairwise, 50% = chance) - indicative, not authoritative."
            )
        if sc and sc.get("score_time"):
            st.caption(f"Scored in {sc['score_time']:.1f}s")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
