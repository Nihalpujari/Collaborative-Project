"""
Multimedia Coherence Score using ImageBind
All 3 modalities encoded in ONE shared embedding space.

Formula:
  s1 = sim(text  <-> image)   via ImageBind
  s2 = sim(text  <-> audio)   via ImageBind
  s3 = sim(image <-> audio)   via ImageBind  (direct — no text bridge)

  TSAS      = (s1 + s2 + s3) / 3
  Coherence = TSAS - lambda * variance(s1, s2, s3)

Tests lambda = 0.1, 0.5, 1.0, 2.0 across all available prompts.
"""

import numpy as np
import torch
import torchaudio
import soundfile as sf
from scipy import signal as scipy_signal
import pandas as pd
from pathlib import Path

from imagebind import data as ib_data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

from prompts import PROMPTS

# ── paths ─────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
TEXT_DIR  = BASE / "outputs/text"
IMAGE_DIR = BASE / "outputs/images"
AUDIO_DIR = BASE / "outputs/audio"
OUT_DIR   = BASE / "outputs/scores"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── lambda values to test ─────────────────────────────────────────────────────
LAMBDAS = [0.1, 0.5, 1.0, 2.0]

# ── device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ── load ImageBind ────────────────────────────────────────────────────────────
print("Loading ImageBind model...")
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(DEVICE)
print("ImageBind loaded.\n")


# ── helpers ───────────────────────────────────────────────────────────────────

def cosine(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


def _load_audio_tensor(audio_path):
    """Load .wav → mel spectrogram tensor expected by ImageBind's audio encoder.
    Avoids TorchCodec; uses soundfile + torchaudio instead."""
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        n = int(len(audio) * 16000 / sr)
        audio = scipy_signal.resample(audio, n).astype("float32")

    waveform = torch.tensor(audio).unsqueeze(0)  # [1, T]

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=128,
        f_min=0.0,
        f_max=8000.0,
    )
    mel = mel_transform(waveform)          # [1, 128, T]
    mel = (mel + 1e-6).log()
    mel = (mel + 4.268) / 4.569           # ImageBind normalisation constants

    target_len = 204                       # 2-second clip at hop_length=160
    t = mel.shape[-1]
    if t < target_len:
        mel = torch.nn.functional.pad(mel, (0, target_len - t))
    else:
        mel = mel[:, :, :target_len]

    return mel.unsqueeze(0)                # [1, 1, 128, 204]


def get_embeddings(text, image_path, audio_path):
    audio_tensor = _load_audio_tensor(audio_path).to(DEVICE)

    inputs = {
        ModalityType.TEXT:   ib_data.load_and_transform_text([text], DEVICE),
        ModalityType.VISION: ib_data.load_and_transform_vision_data([str(image_path)], DEVICE),
        ModalityType.AUDIO:  audio_tensor,
    }
    with torch.no_grad():
        embeddings = model(inputs)

    t = embeddings[ModalityType.TEXT][0].cpu().numpy()
    i = embeddings[ModalityType.VISION][0].cpu().numpy()
    a = embeddings[ModalityType.AUDIO][0].cpu().numpy()
    return t, i, a


# ── score one prompt ──────────────────────────────────────────────────────────

def score_prompt(idx, prompt):
    image_path = IMAGE_DIR / f"prompt_{idx}.png"
    audio_path = AUDIO_DIR / f"prompt_{idx}.wav"
    text_path  = TEXT_DIR  / f"prompt_{idx}.txt"

    if not image_path.exists():
        return None

    generated_text = text_path.read_text(encoding="utf-8").strip()[:200]

    try:
        t_emb, i_emb, a_emb = get_embeddings(generated_text, image_path, audio_path)
    except Exception as e:
        print(f"  [ERROR] Prompt {idx}: {e}")
        return None

    s1 = cosine(t_emb, i_emb)
    s2 = cosine(t_emb, a_emb)
    s3 = cosine(i_emb, a_emb)

    tsas     = (s1 + s2 + s3) / 3
    variance = float(np.var([s1, s2, s3]))

    result = {
        "prompt_id":     idx,
        "prompt":        prompt[:80],
        "s1_text_image": round(s1, 4),
        "s2_text_audio": round(s2, 4),
        "s3_image_audio":round(s3, 4),
        "tsas":          round(tsas, 4),
        "variance":      round(variance, 4),
    }

    for lam in LAMBDAS:
        result[f"score_λ{lam}"] = round(tsas - lam * variance, 4)

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  MULTIMEDIA COHERENCE SCORE — ImageBind — All Prompts")
    print(f"  Lambda values tested: {LAMBDAS}")
    print("=" * 70 + "\n")

    results = []
    for idx, prompt in enumerate(PROMPTS, start=1):
        print(f"  [{idx:02d}/56] {prompt[:60]}...")
        r = score_prompt(idx, prompt)
        if r:
            results.append(r)
            print(f"         s1={r['s1_text_image']}  s2={r['s2_text_audio']}  "
                  f"s3={r['s3_image_audio']}  TSAS={r['tsas']}  var={r['variance']}")
        else:
            print(f"         [SKIPPED]")

    # ── save to CSV ───────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    out_path = OUT_DIR / "imagebind_coherence_scores.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY — Average scores across all prompts")
    print("=" * 70)
    print(f"  {'Metric':<20} {'Average'}")
    print(f"  {'-'*35}")
    print(f"  {'s1 (text↔image)':<20} {df['s1_text_image'].mean():.4f}")
    print(f"  {'s2 (text↔audio)':<20} {df['s2_text_audio'].mean():.4f}")
    print(f"  {'s3 (image↔audio)':<20} {df['s3_image_audio'].mean():.4f}")
    print(f"  {'TSAS':<20} {df['tsas'].mean():.4f}")
    print(f"  {'Variance':<20} {df['variance'].mean():.4f}")
    print()
    for lam in LAMBDAS:
        col = f"score_λ{lam}"
        diff = df['tsas'].mean() - df[col].mean()
        print(f"  Score (λ={lam:<4}) avg = {df[col].mean():.4f}  "
              f"(penalty effect: -{diff:.4f})")

    # ── best and worst prompts ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TOP 3 most coherent prompts (λ=0.5)")
    print("=" * 70)
    top3 = df.nlargest(3, "score_λ0.5")[["prompt_id", "prompt", "tsas", "score_λ0.5"]]
    for _, row in top3.iterrows():
        print(f"  [{int(row['prompt_id'])}] {row['prompt'][:60]}")
        print(f"       TSAS={row['tsas']}  Score={row['score_λ0.5']}\n")

    print("  BOTTOM 3 least coherent prompts (λ=0.5)")
    print("=" * 70)
    bot3 = df.nsmallest(3, "score_λ0.5")[["prompt_id", "prompt", "tsas", "score_λ0.5"]]
    for _, row in bot3.iterrows():
        print(f"  [{int(row['prompt_id'])}] {row['prompt'][:60]}")
        print(f"       TSAS={row['tsas']}  Score={row['score_λ0.5']}\n")


if __name__ == "__main__":
    main()
