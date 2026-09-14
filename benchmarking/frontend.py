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
# ---------------------------
BASE_DIR = Path(os.path.abspath(__file__)).parent
_spec = importlib.util.spec_from_file_location("secrets", BASE_DIR.parent / "secrets.py")
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CF_ACCOUNT_ID = _mod.CLOUDFLARE_ACCOUNT_ID
CF_API_TOKEN  = _mod.CLOUDFLARE_API_TOKEN

CF_BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run"
CF_HEADERS  = {
    "Authorization": f"Bearer {CF_API_TOKEN}",
    "Content-Type": "application/json",
}

# ---------------------------
# Model Options
# ---------------------------
TEXT_MODELS = {
    "Llama 3.1 8B":   "@cf/meta/llama-3.1-8b-instruct",
    "Llama 4 Scout":  "@cf/meta/llama-4-scout-17b-16e-instruct",
    "Kimi K2.6":      "@cf/moonshotai/kimi-k2.6",
    "GPT-OSS 20B":    "@cf/openai/gpt-oss-20b",
}

IMAGE_MODELS = {
    "Flux 1 Schnell":        "@cf/black-forest-labs/flux-1-schnell",
    "Flux 2 Dev":            "@cf/black-forest-labs/flux-2-dev",
    "Leonardo Lucid Origin": "@cf/leonardo/lucid-origin",
    "Leonardo Phoenix 1.0":  "@cf/leonardo/phoenix-1.0",
}

AUDIO_MODELS = {
    "MeloTTS":       "@cf/myshell-ai/melotts",
    "Deepgram Aura-1": "@cf/deepgram/aura-1",
    "Deepgram Aura-2": "@cf/deepgram/aura-2-es",
}


# ---------------------------
# Generation Functions
# ---------------------------
def generate_text(prompt, model_id):
    try:
        r = requests.post(
            f"{CF_BASE_URL}/{model_id}",
            headers=CF_HEADERS,
            json={
                "messages": [{"role": "user", "content": f"Write a vivid, detailed and engaging description of: {prompt}"}],
                "max_tokens": 400,
            },
            timeout=60,
        )
        if r.status_code == 200:
            return r.json().get("result", {}).get("response", ""), None
        return None, f"Error {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, str(e)


def generate_image(prompt, model_id):
    try:
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
            # some models return raw bytes
            if r.content:
                return r.content, None
        return None, f"Error {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, str(e)


def generate_audio(text, model_id):
    try:
        payload = {"prompt": text[:300]}
        # Deepgram models use different payload key
        if "deepgram" in model_id:
            payload = {"text": text[:300]}
        r = requests.post(
            f"{CF_BASE_URL}/{model_id}",
            headers=CF_HEADERS,
            json=payload,
            timeout=60,
        )
        if r.status_code == 200:
            result = r.json().get("result", {})
            audio_b64 = result.get("audio", "")
            if audio_b64:
                return base64.b64decode(audio_b64), None
            # Deepgram may return raw bytes
            if r.content:
                return r.content, None
        return None, f"Error {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, str(e)


# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="AI Content Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
.empty-state { text-align: center; padding: 4rem 1rem; color: #94A3B8; }
.empty-icon  { font-size: 2.8rem; margin-bottom: 0.7rem; }
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
</style>
""", unsafe_allow_html=True)


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

    st.markdown("<div style='height:1px; background:#334155; margin:1rem 0 0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:0.75rem; color:#475569; line-height:1.6;'>"
        f"<b style='color:#94A3B8;'>Text</b><br>{TEXT_MODELS[selected_text_model]}<br><br>"
        f"<b style='color:#94A3B8;'>Image</b><br>{IMAGE_MODELS[selected_image_model]}<br><br>"
        f"<b style='color:#94A3B8;'>Audio</b><br>{AUDIO_MODELS[selected_audio_model]}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------
# Session State
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------
# Page Header
# ---------------------------
st.markdown('<div class="page-title">AI Content Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">One prompt → text · image · audio </div>',
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
    })
    st.rerun()


# ---------------------------
# History
# ---------------------------
if not st.session_state.history:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">✦</div>
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

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
