import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
Stage 1 - Per-Modality Quality Scores

Computes Q_text, Q_image, Q_audio for all available prompts.

Formulas:
  Q_text  = BERTScore(generated_text, prompt)
  Q_image = avg(CLIP_score, Aesthetic_score) - lambda x variance(CLIP_score, Aesthetic_score)
  Q_audio = avg(semantic_score, 1 - WER)     - lambda x variance(semantic_score, 1 - WER)

  semantic_score = BERTScore(Whisper_transcript, prompt)
  -- measures whether the spoken content aligns with the prompt

Saves results to outputs/scores/quality_scores.csv
"""

import numpy as np
import torch
import whisper
import soundfile as sf
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy import signal as scipy_signal
from jiwer import wer as compute_wer

from transformers import (
    CLIPProcessor, CLIPModel,
    pipeline as hf_pipeline,
)
from bert_score import score as bert_score_fn

from prompts import PROMPTS

# -- paths -------------------------------------------------------------------
BASE      = Path(__file__).parent.parent   # cloudflare/ folder where outputs/ lives
TEXT_DIR  = BASE / "outputs/text"
IMAGE_DIR = BASE / "outputs/images"
AUDIO_DIR = BASE / "outputs/audio"
OUT_DIR   = BASE / "outputs/scores"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- lambda ------------------------------------------------------------------
LAMBDA = 0.5

# -- device ------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# -- load models -------------------------------------------------------------
print("Loading CLIP...")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

print("Loading Whisper...")
whisper_model = whisper.load_model("base")

print("Loading Aesthetic Predictor...")
aesthetic_pipe = hf_pipeline(
    "image-classification",
    model="cafeai/cafe_aesthetic",
    device=0 if DEVICE == "cuda" else -1,
)

print("All models loaded.\n")


# -- helpers -----------------------------------------------------------------

def cosine(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


def resample_audio(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    n = int(len(audio) * target_sr / orig_sr)
    return scipy_signal.resample(audio, n).astype("float32")


def quality_pair(a, b, lam=LAMBDA):
    """avg(a, b) - lambda x variance(a, b)  -- penalises imbalance between two scores."""
    avg = (a + b) / 2
    var = float(np.var([a, b]))
    return float(np.clip(avg - lam * var, 0, 1))


# -- 3A - Text Quality -------------------------------------------------------

def get_q_text(generated_text, prompt):
    """BERTScore F1 between generated text and prompt."""
    _, _, F1 = bert_score_fn(
        [generated_text], [prompt],
        lang="en", verbose=False,
    )
    return float(F1.mean())


# -- 3B - Image Quality ------------------------------------------------------

def get_clip_score(prompt, image_path):
    """Cosine similarity between CLIP text embedding of prompt and image embedding."""
    image = Image.open(image_path).convert("RGB")
    text_inputs  = clip_processor(text=[prompt], return_tensors="pt", padding=True)
    image_inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        t_emb = clip_model.get_text_features(**text_inputs).squeeze().numpy()
        i_emb = clip_model.get_image_features(**image_inputs).squeeze().numpy()
    return float(np.clip(cosine(t_emb, i_emb), 0, 1))


def get_aesthetic_score(image_path):
    """LAION aesthetic score - probability the image is 'aesthetic'."""
    result = aesthetic_pipe(str(image_path))
    for r in result:
        if r["label"] == "aesthetic":
            return float(r["score"])
    return 0.0


def get_q_image(prompt, image_path):
    clip_s = get_clip_score(prompt, image_path)
    aes_s  = get_aesthetic_score(image_path)
    return clip_s, aes_s, quality_pair(clip_s, aes_s)


# -- 3C - Audio Quality ------------------------------------------------------

def transcribe_audio(audio_path):
    """Transcribe audio file using Whisper. Returns transcript string."""
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = resample_audio(audio, sr, 16000)
    return whisper_model.transcribe(audio)["text"].strip()


def get_semantic_score(prompt, transcript):
    """BERTScore F1 between Whisper transcript and original prompt.
    Measures whether the spoken content is semantically aligned with the prompt."""
    _, _, F1 = bert_score_fn(
        [transcript], [prompt],
        lang="en", verbose=False,
    )
    return float(F1.mean())


def get_wer_inv(generated_text, transcript):
    """1 - WER between generated text and Whisper transcript. Higher = better TTS accuracy."""
    error = float(np.clip(compute_wer(generated_text.lower(), transcript.lower()), 0, 1))
    return 1 - error


def get_q_audio(prompt, audio_path, generated_text):
    transcript   = transcribe_audio(audio_path)
    semantic_s   = get_semantic_score(prompt, transcript)
    wer_inv      = get_wer_inv(generated_text, transcript)
    return semantic_s, wer_inv, quality_pair(semantic_s, wer_inv), transcript


# -- score one prompt --------------------------------------------------------

def score_prompt(idx, prompt):
    image_path = IMAGE_DIR / f"prompt_{idx}.png"
    audio_path = AUDIO_DIR / f"prompt_{idx}.wav"
    text_path  = TEXT_DIR  / f"prompt_{idx}.txt"

    if not image_path.exists():
        return None

    generated_text = text_path.read_text(encoding="utf-8").strip()

    try:
        q_text                                  = get_q_text(generated_text, prompt)
        clip_s, aes_s, q_image                  = get_q_image(prompt, image_path)
        semantic_s, wer_inv, q_audio, transcript = get_q_audio(prompt, audio_path, generated_text)

        return {
            "prompt_id":      idx,
            "prompt":         prompt[:80],
            "q_text":         round(q_text,      4),
            "clip_score":     round(clip_s,      4),
            "aesthetic":      round(aes_s,       4),
            "q_image":        round(q_image,     4),
            "semantic_score": round(semantic_s,  4),
            "wer_inv":        round(wer_inv,     4),
            "q_audio":        round(q_audio,     4),
            "transcript":     transcript[:120],
        }

    except Exception as e:
        print(f"  [ERROR] Prompt {idx}: {e}")
        return None


# -- main --------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  STAGE 1 - Per-Modality Quality Scores")
    print(f"  lambda = {LAMBDA}")
    print("=" * 70 + "\n")

    results = []
    for idx, prompt in enumerate(PROMPTS, start=1):
        print(f"  [{idx:02d}/56] {prompt[:55]}...")
        r = score_prompt(idx, prompt)
        if r:
            results.append(r)
            print(f"         Q_text={r['q_text']}  "
                  f"Q_image={r['q_image']} (CLIP={r['clip_score']} Aes={r['aesthetic']})  "
                  f"Q_audio={r['q_audio']} (Sem={r['semantic_score']} WER_inv={r['wer_inv']})")
        else:
            print(f"         [SKIPPED - no image]")

    df = pd.DataFrame(results)
    out_path = OUT_DIR / "quality_scores.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Prompts scored : {len(results)}")
    print(f"  {'Metric':<20} {'Average':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*48}")
    for col, label in [("q_text", "Q_text"), ("q_image", "Q_image"), ("q_audio", "Q_audio")]:
        print(f"  {label:<20} {df[col].mean():>8.4f}  {df[col].min():>8.4f}  {df[col].max():>8.4f}")

    print("\n  TOP 3 best overall quality (avg of Q_text, Q_image, Q_audio):")
    df["q_overall"] = (df["q_text"] + df["q_image"] + df["q_audio"]) / 3
    top3 = df.nlargest(3, "q_overall")[["prompt_id", "prompt", "q_text", "q_image", "q_audio", "q_overall"]]
    for _, row in top3.iterrows():
        print(f"    [{int(row['prompt_id'])}] {row['prompt'][:55]}")
        print(f"         Q_text={row['q_text']}  Q_image={row['q_image']}  Q_audio={row['q_audio']}  Overall={row['q_overall']:.4f}\n")

    print("  BOTTOM 3 worst overall quality:")
    bot3 = df.nsmallest(3, "q_overall")[["prompt_id", "prompt", "q_text", "q_image", "q_audio", "q_overall"]]
    for _, row in bot3.iterrows():
        print(f"    [{int(row['prompt_id'])}] {row['prompt'][:55]}")
        print(f"         Q_text={row['q_text']}  Q_image={row['q_image']}  Q_audio={row['q_audio']}  Overall={row['q_overall']:.4f}\n")


if __name__ == "__main__":
    main()
