"""
LR Scorer — real-time scoring using the Likelihood Ratio approach.

Lightweight version: no ImageBind, BERTScore, or Whisper required.
Uses sentence-transformers (CPU-friendly) for text quality,
and CLIP for image-text similarity if available.

Run:
    pip install fastapi uvicorn python-multipart sentence-transformers Pillow
    uvicorn lr_scorer:app --port 8000
"""

from __future__ import annotations
import math, logging, tempfile
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Fitted parameters (trained on 500 prompts, LOO cross-validated) ───────────
W1, W2, W3 = 7.8666, 2.2790, 0.7796   # quality weights: text, image, audio

PARAMS = {
    "f1": dict(mu_p=0.3062, sig_p=0.0344, mu_n=0.2902, sig_n=0.0358),
    "f2": dict(mu_p=0.0545, sig_p=0.0416, mu_n=0.0532, sig_n=0.0421),
    "f3": dict(mu_p=0.0487, sig_p=0.0274, mu_n=0.0468, sig_n=0.0280),
}

JUDGE_THRESHOLD = 3.5


# ── lazy model cache ──────────────────────────────────────────────────────────
_st_model  = None   # sentence-transformers
_clip      = None   # CLIP (optional)

def _get_st():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer, util
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("SentenceTransformer loaded")
    return _st_model

def _get_clip():
    global _clip
    if _clip is None:
        import clip, torch
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        _clip = (model, preprocess, device)
        log.info("CLIP loaded on %s", device)
    return _clip


# ── quality & similarity ──────────────────────────────────────────────────────

def q_text(prompt: str, text: str) -> float:
    """Cosine similarity between prompt and generated text embeddings."""
    try:
        from sentence_transformers import util
        model = _get_st()
        e1 = model.encode(prompt, convert_to_tensor=True)
        e2 = model.encode(text,   convert_to_tensor=True)
        return float(util.cos_sim(e1, e2).item())
    except Exception as e:
        log.warning("q_text failed: %s", e)
        return 0.5

def q_image_and_s1(prompt: str, image_path: Path) -> tuple[float, float]:
    """
    Returns (Q_image, s1).
    Uses CLIP if available; falls back to dataset means.
    """
    try:
        import torch
        from PIL import Image
        model, preprocess, device = _get_clip()

        img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        import clip
        tokens = clip.tokenize([prompt[:77]]).to(device)

        with torch.no_grad():
            img_feat  = model.encode_image(img)
            txt_feat  = model.encode_text(tokens)
            img_feat  = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat  = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            s1        = float((img_feat * txt_feat).sum().item())

        # CLIP score as quality proxy (higher similarity → higher quality)
        q_img = min(max((s1 + 1) / 2, 0.0), 1.0)
        return q_img, s1

    except Exception as e:
        log.warning("CLIP failed (%s) — using dataset means", e)
        return 0.472, 0.396   # dataset means from 500-prompt training set

def q_audio() -> float:
    """Audio quality — returns dataset mean (Whisper not required)."""
    return 0.75   # reasonable default; install Whisper for real scoring

def s2_s3() -> tuple[float, float]:
    """Cross-modal similarities involving audio — dataset means."""
    return 0.064, 0.084   # from 500-prompt training set


# ── LR math ──────────────────────────────────────────────────────────────────

def _log_prob(x: float, mu: float, sig: float) -> float:
    return -math.log(sig) - 0.5 * ((x - mu) / sig) ** 2

def _lr_score(f1: float, f2: float, f3: float) -> float:
    total = 0.0
    for f, key in [(f1, "f1"), (f2, "f2"), (f3, "f3")]:
        p = PARAMS[key]
        total += _log_prob(f, p["mu_p"], p["sig_p"]) \
               - _log_prob(f, p["mu_n"], p["sig_n"])
    return total


# ── FastAPI ──────────────────────────────────────────────────────────────────

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import traceback

app = FastAPI(title="LR Scorer API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/score")
async def score_endpoint(
    prompt: str        = Form(...),
    text:   str        = Form(...),
    image:  UploadFile = File(...),
    audio:  UploadFile = File(...),
):
    try:
        suffix_img = Path(image.filename or "image.png").suffix or ".png"

        with tempfile.NamedTemporaryFile(suffix=suffix_img, delete=False) as fi:
            fi.write(await image.read())
            img_path = Path(fi.name)

        # discard audio bytes — we use dataset means for audio features
        await audio.read()

        # quality scores
        Qt  = q_text(prompt, text)
        Qi, s1 = q_image_and_s1(prompt, img_path)
        Qa  = q_audio()
        s2, s3 = s2_s3()

        img_path.unlink(missing_ok=True)

        # weighted features
        f1 = s1 * (W1 * Qt + W2 * Qi) / (W1 + W2)
        f2 = s2 * (W1 * Qt + W3 * Qa) / (W1 + W3)
        f3 = s3 * (W2 * Qi + W3 * Qa) / (W2 + W3)

        score = _lr_score(f1, f2, f3)
        label = "Good" if score > 0 else "Not-Good"

        return {
            "score":    round(score, 6),
            "label":    label,
            "features": {"f1": round(f1, 6), "f2": round(f2, 6), "f3": round(f3, 6)},
            "quality":  {
                "Q_text":  round(Qt,  4),
                "Q_image": round(Qi,  4),
                "Q_audio": round(Qa,  4),
                "s1": round(s1, 6),
                "s2": round(s2, 6),
                "s3": round(s3, 6),
            },
        }

    except Exception:
        log.error("Scoring failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get("/params")
def get_params():
    return {"weights": {"w1": W1, "w2": W2, "w3": W3}, "gaussians": PARAMS}
