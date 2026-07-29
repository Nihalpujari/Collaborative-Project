"""
Multimedia Coherence Score — test on 5 Cloudflare outputs

Formulas:
  s1 = sim(text  <-> image)   via CLIP
  s2 = sim(text  <-> audio)   via CLAP
  s3 = sim(image <-> audio)   via CLIP (audio transcribed by Whisper first)

  TSAS  = (s1 + s2 + s3) / 3
  score = TSAS - lambda * variance(s1, s2, s3)
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from scipy import signal as scipy_signal
from PIL import Image
import torch
import whisper
from transformers import (
    CLIPProcessor, CLIPModel,
    ClapProcessor, ClapModel,
)

# ── paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
TEXT_DIR  = BASE / "outputs/text"
IMAGE_DIR = BASE / "outputs/images"
AUDIO_DIR = BASE / "outputs/audio"

# ── prompts (first 5) ────────────────────────────────────────────────────────
from prompts import PROMPTS
PROMPTS_5 = PROMPTS[:5]

# ── lambda (penalty weight) ──────────────────────────────────────────────────
LAMBDA = 0.5

# ── load models (once) ──────────────────────────────────────────────────────
print("Loading models...")

clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

clap_model     = ClapModel.from_pretrained("laion/clap-htsat-unfused")
clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

whisper_model  = whisper.load_model("base")

print("Models loaded.\n")


# ── helpers ──────────────────────────────────────────────────────────────────

def cosine(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


def resample(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    n = int(len(audio) * target_sr / orig_sr)
    return scipy_signal.resample(audio, n).astype("float32")


def clip_text_embed(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        return clip_model.get_text_features(**inputs).squeeze().numpy()


def clip_image_embed(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        return clip_model.get_image_features(**inputs).squeeze().numpy()


def clap_text_embed(text):
    inputs = clap_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        return clap_model.get_text_features(**inputs).squeeze().numpy()


def clap_audio_embed(audio_path):
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = resample(audio, sr, 48000)
    inputs = clap_processor(
        audios=[audio], return_tensors="pt",
        padding=True, sampling_rate=48000
    )
    with torch.no_grad():
        return clap_model.get_audio_features(**inputs).squeeze().numpy()


def transcribe(audio_path):
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = resample(audio, sr, 16000)
    result = whisper_model.transcribe(audio)
    return result["text"].strip()


# ── scoring ──────────────────────────────────────────────────────────────────

def score_prompt(idx, prompt):
    print(f"  Prompt {idx}: \"{prompt[:60]}...\"" if len(prompt) > 60 else f"  Prompt {idx}: \"{prompt}\"")

    text_path  = TEXT_DIR  / f"prompt_{idx}.txt"
    image_path = IMAGE_DIR / f"prompt_{idx}.png"
    audio_path = AUDIO_DIR / f"prompt_{idx}.wav"

    if not image_path.exists():
        print(f"    [SKIP] No image found for prompt {idx}\n")
        return None

    # read generated text (use the generated text, not the prompt)
    generated_text = text_path.read_text(encoding="utf-8").strip()

    # s1: sim(text <-> image)
    t_embed = clip_text_embed(generated_text[:200])   # CLIP has token limit
    i_embed = clip_image_embed(image_path)
    s1 = cosine(t_embed, i_embed)

    # s2: sim(text <-> audio)
    ct_embed = clap_text_embed(prompt)
    ca_embed = clap_audio_embed(audio_path)
    s2 = cosine(ct_embed, ca_embed)

    # s3: sim(image <-> audio) via Whisper transcript -> CLIP
    transcript = transcribe(audio_path)
    tr_embed = clip_text_embed(transcript[:200])
    s3 = cosine(tr_embed, i_embed)

    # TSAS
    tsas = (s1 + s2 + s3) / 3

    # variance penalty
    variance = np.var([s1, s2, s3])
    score = tsas - LAMBDA * variance

    print(f"    s1 (text <-> image):  {s1:.4f}")
    print(f"    s2 (text <-> audio):  {s2:.4f}")
    print(f"    s3 (image <-> audio): {s3:.4f}")
    print(f"    TSAS:                 {tsas:.4f}")
    print(f"    Variance:             {variance:.4f}")
    print(f"    Final Score (λ=0.5):  {score:.4f}")
    print()

    return {
        "prompt_id": idx,
        "prompt": prompt,
        "s1_text_image": round(s1, 4),
        "s2_text_audio": round(s2, 4),
        "s3_image_audio": round(s3, 4),
        "tsas": round(tsas, 4),
        "variance": round(variance, 4),
        "final_score": round(score, 4),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  MULTIMEDIA COHERENCE SCORE — 5 PROMPT TEST")
    print(f"  λ (lambda) = {LAMBDA}")
    print("=" * 60 + "\n")

    results = []
    for i, prompt in enumerate(PROMPTS_5, start=1):
        result = score_prompt(i, prompt)
        if result:
            results.append(result)

    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'ID':<4} {'TSAS':<8} {'Variance':<10} {'Final Score'}")
    print(f"  {'-'*40}")
    for r in results:
        print(f"  {r['prompt_id']:<4} {r['tsas']:<8} {r['variance']:<10} {r['final_score']}")

    if results:
        avg_tsas  = np.mean([r["tsas"] for r in results])
        avg_score = np.mean([r["final_score"] for r in results])
        print(f"\n  Average TSAS:        {avg_tsas:.4f}")
        print(f"  Average Final Score: {avg_score:.4f}")
        print(f"  Difference (penalty effect): {avg_tsas - avg_score:.4f}")


if __name__ == "__main__":
    main()
