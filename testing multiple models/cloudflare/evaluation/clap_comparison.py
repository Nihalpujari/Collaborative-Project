"""
CLAP vs Semantic Score Comparison

Computes CLAP scores for all 53 prompts and saves a side-by-side comparison
with semantic_score (BERTScore on Whisper transcript) to show why CLAP was
not suitable for evaluating TTS speech audio.

CLAP (laion/clap-htsat-unfused) was trained on environmental sounds and music.
When given TTS speech audio, it returns near-zero scores because it cannot
understand spoken language — it looks for sound events, not spoken words.

Saves: outputs/scores/clap_comparison.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pandas as pd
import soundfile as sf
from scipy import signal as scipy_signal

from transformers import ClapProcessor, ClapModel
from prompts import PROMPTS

# -- paths -------------------------------------------------------------------
BASE      = Path(__file__).parent.parent
AUDIO_DIR = BASE / "outputs/audio"
OUT_DIR   = BASE / "outputs/scores"
SCORES_CSV = OUT_DIR / "quality_scores.csv"

# -- device ------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# -- load CLAP ---------------------------------------------------------------
print("Loading CLAP (laion/clap-htsat-unfused)...")
clap_model     = ClapModel.from_pretrained("laion/clap-htsat-unfused")
clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
clap_model.eval()
print("CLAP loaded.\n")


# -- helpers -----------------------------------------------------------------

def resample_audio(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    n = int(len(audio) * target_sr / orig_sr)
    return scipy_signal.resample(audio, n).astype("float32")


def cosine(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


def get_clap_score(prompt, audio_path):
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = resample_audio(audio, sr, 48000)
    inputs = clap_processor(
        text=[prompt], audios=[audio],
        return_tensors="pt", padding=True, sampling_rate=48000,
    )
    with torch.no_grad():
        t_emb = clap_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        ).squeeze().numpy()
        a_emb = clap_model.get_audio_features(
            input_features=inputs["input_features"],
        ).squeeze().numpy()
    return float(np.clip(cosine(t_emb, a_emb), 0, 1))


# -- main --------------------------------------------------------------------

def main():
    # Load existing quality_scores.csv to get semantic_score and wer_inv
    if not SCORES_CSV.exists():
        print(f"ERROR: {SCORES_CSV} not found. Run quality_scores.py first.")
        return

    existing = pd.read_csv(SCORES_CSV)[["prompt_id", "prompt", "semantic_score", "wer_inv", "q_audio"]]

    print("=" * 70)
    print("  CLAP vs Semantic Score Comparison")
    print("  Audio type: TTS speech (narration of generated text)")
    print("  CLAP model: laion/clap-htsat-unfused (trained on sounds/music)")
    print("=" * 70 + "\n")

    rows = []
    for _, row in existing.iterrows():
        idx   = int(row["prompt_id"])
        prompt = PROMPTS[idx - 1]
        audio_path = AUDIO_DIR / f"prompt_{idx}.wav"

        print(f"  [{idx:02d}] {prompt[:55]}...")

        clap_s = get_clap_score(prompt, audio_path)
        sem_s  = float(row["semantic_score"])
        wer_inv = float(row["wer_inv"])

        verdict = "CLAP=0 (speech not recognized)" if clap_s < 0.05 else "CLAP has some signal"

        print(f"        CLAP={clap_s:.4f}   Semantic={sem_s:.4f}   WER_inv={wer_inv:.4f}   [{verdict}]")

        rows.append({
            "prompt_id":      idx,
            "prompt":         prompt[:80],
            "clap_score":     round(clap_s,  4),
            "semantic_score": round(sem_s,   4),
            "wer_inv":        round(wer_inv, 4),
            "q_audio_clap":   round(max((clap_s + wer_inv) / 2 - 0.5 * float(np.var([clap_s, wer_inv])), 0), 4),
            "q_audio_sem":    round(float(row["q_audio"]), 4),
            "clap_is_zero":   clap_s < 0.05,
        })

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "clap_comparison.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved to: {out_path}")

    # -- summary stats --------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    n_zero = df["clap_is_zero"].sum()
    total  = len(df)
    print(f"\n  Total prompts scored       : {total}")
    print(f"  Prompts with CLAP < 0.05  : {n_zero} / {total}  ({100*n_zero/total:.0f}%)")
    print(f"\n  {'Metric':<25} {'Mean':>8}  {'Min':>8}  {'Max':>8}  {'Std':>8}")
    print(f"  {'-'*57}")
    for col, label in [
        ("clap_score",     "CLAP score"),
        ("semantic_score", "Semantic score (ours)"),
        ("wer_inv",        "WER_inv"),
        ("q_audio_clap",   "Q_audio with CLAP"),
        ("q_audio_sem",    "Q_audio with Semantic"),
    ]:
        print(f"  {label:<25} {df[col].mean():>8.4f}  {df[col].min():>8.4f}  {df[col].max():>8.4f}  {df[col].std():>8.4f}")

    print(f"\n  CONCLUSION:")
    print(f"  CLAP returns 0.0 for {n_zero}/{total} prompts ({100*n_zero/total:.0f}%) because")
    print(f"  it was trained on environmental sounds and music, not TTS speech.")
    print(f"  Semantic score (BERTScore on Whisper transcript) provides meaningful")
    print(f"  signal across all prompts with std={df['semantic_score'].std():.4f} vs CLAP std={df['clap_score'].std():.4f}.")


if __name__ == "__main__":
    main()
