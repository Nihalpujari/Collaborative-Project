import sys
import os
import base64
import time
import io
import importlib.util
from pathlib import Path
import streamlit as st
import requests

# ---------------------------
# Load credentials
# ---------------------------
from openai import OpenAI  # for Groq/Cerebras (OpenAI-compatible)
from google import genai
from google.genai import types
from huggingface_hub import InferenceClient

BASE_DIR = Path(os.path.abspath(__file__)).parent
_spec = importlib.util.spec_from_file_location("api_keys", BASE_DIR / "api_keys.py")
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Cloudflare
CF_ACCOUNT_ID = _mod.CLOUDFLARE_ACCOUNT_ID
CF_API_TOKEN  = _mod.CLOUDFLARE_API_TOKEN
CF_BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run"
CF_HEADERS  = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}

# Groq (OpenAI-compatible)
groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=_mod.GROQ_KEY)

# Cerebras (OpenAI-compatible)
# cerebras_client = OpenAI(base_url="https://api.cerebras.ai/v1", api_key=_mod.CEREBRAS_KEY)

# Gemini
gemini_client = genai.Client(api_key=_mod.GEMINI_API_KEY)

# Hugging Face
HF_TOKEN = _mod.HF_TOKEN
hf_client = InferenceClient(token=HF_TOKEN)


# ---------------------------
# Text Generation (routed by provider)
# ---------------------------
def generate_text(prompt, model_info):
    provider = model_info["provider"]
    model_id = model_info["id"]
    full_prompt = f"Write a vivid, detailed and engaging description of: {prompt}"

    try:
        if provider == "cloudflare":
            r = requests.post(
                f"{CF_BASE_URL}/{model_id}",
                headers=CF_HEADERS,
                json={"messages": [{"role": "user", "content": full_prompt}], "max_tokens": 200},
                timeout=60,
            )
            if r.status_code == 200:
                return r.json().get("result", {}).get("response", ""), None
            return None, f"Error {r.status_code}: {r.text[:250]}"

        elif provider == "groq":
            resp = groq_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=200,
            )
            return resp.choices[0].message.content, None

        elif provider == "cerebras":
            resp = cerebras_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=200,
            )
            return resp.choices[0].message.content, None

        elif provider == "gemini":
            resp = gemini_client.models.generate_content(
                model=model_id,
                contents=full_prompt,
            )
            return resp.text, None

        elif provider == "huggingface":
            resp = hf_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=200,
            )
            return resp.choices[0].message.content, None

        return None, f"Unknown provider: {provider}"
    except Exception as e:
        return None, str(e)


# ---------------------------
# Image Generation
# ---------------------------
def generate_image(prompt, model_info):
    provider = model_info["provider"]
    model_id = model_info["id"]

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
            return None, f"Error {r.status_code}: {r.text[:250]}"

        elif provider == "gemini":
            resp = gemini_client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
            )
            for part in resp.candidates[0].content.parts:
                if part.inline_data:
                    return part.inline_data.data, None
            return None, "No image in response"

        elif provider == "huggingface":
            pil_img = hf_client.text_to_image(prompt, model=model_id)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return buf.getvalue(), None

        return None, f"Unknown provider: {provider}"
    except Exception as e:
        return None, str(e)


# ---------------------------
# Audio Generation
# ---------------------------
def generate_audio(text, model_info):
    provider = model_info["provider"]
    model_id = model_info["id"]

    import re
    clean = re.sub(r'\s+', ' ', text).strip()
    clean = re.sub(r'[^\w\s.,!?-]', '', clean)

    LIMIT = 250
    if len(clean) > LIMIT:
        window = clean[:LIMIT]
        cut = max(window.rfind('.'), window.rfind('!'), window.rfind('?'))
        clean = window[:cut + 1] if cut != -1 else window

    try:
        if provider == "cloudflare":
            if "deepgram" in model_id:
                payload = {"text": clean}
            elif "melotts" in model_id:
                payload = {"prompt": clean, "lang": "en"}
            else:
                payload = {"prompt": clean}

            last_err = None
            for attempt in range(3):
                r = requests.post(f"{CF_BASE_URL}/{model_id}", headers=CF_HEADERS, json=payload, timeout=60)
                if r.status_code == 200:
                    try:
                        audio_b64 = r.json().get("result", {}).get("audio", "")
                    except ValueError:
                        audio_b64 = ""
                    if audio_b64:
                        return base64.b64decode(audio_b64), None
                    if r.content:
                        return r.content, None
                last_err = f"Error {r.status_code}: {r.text[:250]}"
                if attempt < 2:
                    time.sleep(1.5)
            return None, last_err

        elif provider == "gemini":
            resp = gemini_client.models.generate_content(
                model=model_id,
                contents=clean,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
                        )
                    ),
                ),
            )
            for part in resp.candidates[0].content.parts:
                if part.inline_data:
                    return part.inline_data.data, None
            return None, "No audio in response"

        elif provider == "huggingface":
            audio_bytes = hf_client.text_to_speech(clean, model=model_id)
            return audio_bytes, None

        return None, f"Unknown provider: {provider}"
    except Exception as e:
        return None, str(e)


# ---------------------------
# Model Options
# ---------------------------
TEXT_MODELS = {
    "Llama 3.1 8B (Cloudflare)":   {"provider": "cloudflare", "id": "@cf/meta/llama-3.1-8b-instruct"},
    "Llama 4 Scout (Cloudflare)":  {"provider": "cloudflare", "id": "@cf/meta/llama-4-scout-17b-16e-instruct"},
    "Kimi K2.6 (Cloudflare)":      {"provider": "cloudflare", "id": "@cf/moonshotai/kimi-k2.6"},
    "GPT-OSS 20B (Cloudflare)":    {"provider": "cloudflare", "id": "@cf/openai/gpt-oss-20b"},
    "Llama 3.3 70B (Groq)":        {"provider": "groq", "id": "llama-3.3-70b-versatile"},
    "Llama 3.1 8B Instant (Groq)": {"provider": "groq", "id": "llama-3.1-8b-instant"},
    "Gemini 2.5 Flash (Google)":   {"provider": "gemini", "id": "gemini-2.5-flash"},
    "Llama 3.1 8B (HuggingFace)":  {"provider": "huggingface", "id": "meta-llama/Llama-3.1-8B-Instruct"},
    "Zephyr 7B (HuggingFace)":     {"provider": "huggingface", "id": "HuggingFaceH4/zephyr-7b-beta"},
}

IMAGE_MODELS = {
    "Flux 1 Schnell (Cloudflare)":        {"provider": "cloudflare", "id": "@cf/black-forest-labs/flux-1-schnell"},
    "Flux 2 Dev (Cloudflare)":            {"provider": "cloudflare", "id": "@cf/black-forest-labs/flux-2-dev"},
    "Leonardo Lucid Origin (Cloudflare)": {"provider": "cloudflare", "id": "@cf/leonardo/lucid-origin"},
    "Leonardo Phoenix 1.0 (Cloudflare)":  {"provider": "cloudflare", "id": "@cf/leonardo/phoenix-1.0"},
    "Gemini Flash Image (Google)":        {"provider": "gemini", "id": "gemini-2.5-flash-image"},
    "Flux 1 Schnell (HuggingFace)":       {"provider": "huggingface", "id": "black-forest-labs/FLUX.1-schnell"},
    "Stable Diffusion XL (HuggingFace)":  {"provider": "huggingface", "id": "stabilityai/stable-diffusion-xl-base-1.0"},
}

AUDIO_MODELS = {
    "MeloTTS (Cloudflare)":         {"provider": "cloudflare", "id": "@cf/myshell-ai/melotts"},
    "Deepgram Aura-1 (Cloudflare)": {"provider": "cloudflare", "id": "@cf/deepgram/aura-1"},
    "Deepgram Aura-2 (Cloudflare)": {"provider": "cloudflare", "id": "@cf/deepgram/aura-2-es"},
    "Gemini TTS (Google)":          {"provider": "gemini", "id": "gemini-2.5-flash-preview-tts"},
    "Facebook MMS-TTS (HuggingFace)": {"provider": "huggingface", "id": "facebook/mms-tts-eng"},
}


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
# change the filename here to restyle — nothing else needs touching.
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

.page-title { font-size: 2rem; font-weight: 800; color: #0F172A; }
.page-subtitle { font-size: 0.95rem; color: #64748B; margin-top: 0.3rem; margin-bottom: 1.5rem; }

/* Hero banner — falls back to a flat gradient if the artwork is missing */
.hero {
    position: relative;
    border-radius: 18px;
    overflow: hidden;
    padding: 2.1rem 2.2rem;
    margin-bottom: 1.8rem;
    background: linear-gradient(100deg, #0B1526 0%, #16294A 60%, #1E3A6B 100%);
    box-shadow: 0 8px 28px rgba(15,23,42,0.18);
}
.hero-badge {
    display: inline-block;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #93C5FD;
    border: 1px solid rgba(147,197,253,0.35);
    border-radius: 999px;
    padding: 0.24rem 0.7rem;
    margin-bottom: 0.85rem;
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
    margin-bottom: 0.6rem;
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
.stage-flow {
    font-size: 0.8rem;
    color: #64748B;
    margin-bottom: 1.2rem;
    font-weight: 600;
}
.divider {
    height: 1px;
    background: linear-gradient(to right, #CBD5E1, transparent);
    margin: 1.5rem 0 2rem 0;
}
.empty-state { text-align: center; padding: 2.2rem 1rem 3rem; color: #94A3B8; }
.empty-icon  { font-size: 2.8rem; margin-bottom: 0.7rem; }
.empty-text  { font-size: 1rem; font-weight: 500; }
.empty-art {
    display: block;
    width: 100%;
    max-width: 540px;
    margin: 0 auto 1.5rem;
    border-radius: 16px;
    box-shadow: 0 12px 34px rgba(15,23,42,0.16);
}
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

/* Sidebar — same artwork cropped to the bulb, fading to solid at the
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
# Sidebar — Model Selection
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
    <div class="hero-sub">Chained pipeline → text → image → audio</div>
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
        placeholder="Describe a scene, topic, or idea — e.g. A mountain lake surrounded by pine trees at sunrise",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("✦  Run chained pipeline", type="primary")
st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------
# Generation — CHAINED PIPELINE
# prompt -> text -> image -> audio
# ---------------------------
if submitted and user_input.strip():
    text_out = image_out = audio_out = None
    text_err = image_err = audio_err = None
    text_time = image_time = audio_time = 0.0

    text_model_id  = TEXT_MODELS[selected_text_model]
    image_model_id = IMAGE_MODELS[selected_image_model]
    audio_model_id = AUDIO_MODELS[selected_audio_model]

    progress = st.progress(0, text="Starting pipeline...")

    # STEP 1 — TEXT
    progress.progress(10, text="Step 1/3 — Generating text...")
    t0 = time.perf_counter()
    text_out, text_err = generate_text(user_input, text_model_id)
    text_time = time.perf_counter() - t0

    # STEP 2 — IMAGE (from the generated text)
    progress.progress(40, text="Step 2/3 — Generating image...")
    t1 = time.perf_counter()
    image_source = text_out or user_input
    image_out, image_err = generate_image(image_source, image_model_id)
    image_time = time.perf_counter() - t1

    # STEP 3 — AUDIO (narrates the generated text)
    progress.progress(75, text="Step 3/3 — Generating audio...")
    t3 = time.perf_counter()
    audio_source = text_out or user_input
    audio_out, audio_err = generate_audio(audio_source, audio_model_id)
    audio_time = time.perf_counter() - t3

    progress.progress(100, text="Done.")
    time.sleep(0.3)
    progress.empty()

    st.session_state.history.append({
        "prompt":       user_input.strip(),
        "text":         text_out,
        "text_err":     text_err,
        "text_time":    text_time,
        "text_model":   selected_text_model,
        "image":        image_out,
        "image_err":    image_err,
        "image_time":   image_time,
        "image_model":  selected_image_model,
        "audio":        audio_out,
        "audio_err":    audio_err,
        "audio_time":   audio_time,
        "audio_model":  selected_audio_model,
    })
    st.rerun()


# ---------------------------
# History
# ---------------------------
if not st.session_state.history:
    art = f'<img class="empty-art" src="{EMPTY_URI}" alt="">' if EMPTY_URI else '<div class="empty-icon">✦</div>'
    st.markdown(f"""
    <div class="empty-state">
        {art}
        <div class="empty-text">Enter a prompt to generate text, image and audio</div>
    </div>
    """, unsafe_allow_html=True)

for entry in reversed(st.session_state.history):
    st.markdown(f"<div class='prompt-chip'>💬 {entry['prompt']}</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='stage-flow'>🔗 Chain:&nbsp; Prompt → Text → Image → Audio</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f'<div class="card-label label-text">1 · Text — {entry.get("text_model","")}</div>', unsafe_allow_html=True)
        if entry.get("text_err"):
            st.error(entry["text_err"])
        elif entry.get("text"):
            st.markdown(
                f"<div style='color:#1E293B; font-size:0.9rem; line-height:1.55;'>{entry['text']}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Generated in {entry['text_time']:.1f}s")

    with c2:
        st.markdown(f'<div class="card-label label-image">2 · Image — {entry.get("image_model","")}</div>', unsafe_allow_html=True)
        if entry.get("image_err"):
            st.error(entry["image_err"])
        elif entry.get("image"):
            st.image(entry["image"], width="stretch")
            st.caption(f"From the text · {entry['image_time']:.1f}s")

    with c3:
        st.markdown(f'<div class="card-label label-audio">3 · Audio — {entry.get("audio_model","")}</div>', unsafe_allow_html=True)
        if entry.get("audio_err"):
            st.error(entry["audio_err"])
        elif entry.get("audio"):
            st.audio(entry["audio"], format="audio/wav")
            st.caption(f"Generated in {entry['audio_time']:.1f}s")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
