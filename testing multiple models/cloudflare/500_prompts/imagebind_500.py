"""
imagebind_500.py
================
Computes ImageBind coherence scores (s1, s2, s3) for 500 prompts.
Only processes prompt_ids that have BOTH image AND audio files.

s1 = cosine_sim(text, image)
s2 = cosine_sim(text, audio)
s3 = cosine_sim(image, audio)
TSAS = (s1 + s2 + s3) / 3

Then computes WTSAS using weights learned from 500-prompt linear regression.

Input:
  raw_scores_500.csv                     -> Q features + prompt text
  judge_gemini_gemini-3.1-flash-lite.csv -> LLM ratings
  outputs500/images/prompt_N.png
  outputs500/audio/prompt_N.wav

Output:
  scores/imagebind_500.csv     -> s1, s2, s3, tsas per prompt
  scores/wtsas_500.csv         -> full WTSAS results + comparison
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root for imagebind

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from scipy import signal as scipy_signal

from imagebind import data as ib_data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


def _load_audio_tensor(audio_path):
    """Load .wav → mel spectrogram tensor for ImageBind's audio encoder.
    Uses soundfile + torchaudio to avoid the TorchCodec dependency."""
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

HERE      = Path(__file__).parent
OUTPUTS   = Path(r"C:\Users\hp\Downloads\outputs500-20260910T040634Z-1-001\outputs500")
IMAGE_DIR = OUTPUTS / "images"
AUDIO_DIR = OUTPUTS / "audio"
TEXT_DIR  = OUTPUTS / "text"
RAW       = Path(r"C:\Users\hp\OneDrive\Desktop\raw_scores_500.csv")
JUDGE     = Path(r"C:\Users\hp\OneDrive\Desktop\judge_gemini_gemini-3.1-flash-lite.csv")
IB_CSV    = HERE / "scores" / "imagebind_500.csv"
OUT_CSV   = HERE / "scores" / "wtsas_500.csv"
CKPT      = HERE / "scores" / "imagebind_500_checkpoint.csv"
HERE.joinpath("scores").mkdir(parents=True, exist_ok=True)

# Weights learned from 500-prompt linear regression (run_all_500.py)
W1, W2, W3 = 7.8666, 2.2790, 0.7796
W_SUM      = W1 + W2 + W3
OPT_LAM    = 1.8   # optimal lambda from 53-prompt sweep

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── quality_pair ─────────────────────────────────────────────────────────────
def quality_pair(a, b, lam=0.5):
    avg = (a + b) / 2.0
    var = ((a - avg) ** 2 + (b - avg) ** 2) / 2.0
    return avg - lam * var

# ── load data ────────────────────────────────────────────────────────────────
print("=" * 65)
print("  Loading data")
print("=" * 65)

raw   = pd.read_csv(RAW)
judge = pd.read_csv(JUDGE)

raw["q_text"]  = raw["bertscore"]
raw["q_image"] = quality_pair(raw["clip"],     raw["aesthetic"])
raw["q_audio"] = quality_pair(raw["semantic"], raw["wer_inv"])

df = pd.merge(raw, judge[["prompt_id","judge_mean","judge_grade"]], on="prompt_id")

# Find prompt_ids with both image AND audio
has_img = {int(p.stem.replace("prompt_","")) for p in IMAGE_DIR.glob("prompt_*.png")}
has_aud = {int(p.stem.replace("prompt_","")) for p in AUDIO_DIR.glob("prompt_*.wav")}
valid   = sorted(has_img & has_aud)
print(f"  Images: {len(has_img)}   Audio: {len(has_aud)}   Both: {len(valid)}")

df_valid = df[df["prompt_id"].isin(valid)].reset_index(drop=True)
print(f"  Processing {len(df_valid)} prompts\n")

# ── load checkpoint if exists ─────────────────────────────────────────────────
done_ids = set()
ckpt_rows = []
if CKPT.exists():
    ckpt_df  = pd.read_csv(CKPT)
    done_ids = set(ckpt_df["prompt_id"].tolist())
    ckpt_rows = ckpt_df.to_dict("records")
    print(f"  Checkpoint found: {len(done_ids)} already done, resuming...\n")

# ── load ImageBind ────────────────────────────────────────────────────────────
print("  Loading ImageBind model (this takes ~30s)...")
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(DEVICE)
print(f"  ImageBind loaded on {DEVICE}\n")

# ── process each prompt ───────────────────────────────────────────────────────
results = list(ckpt_rows)
n = len(df_valid)

for i, row in df_valid.iterrows():
    pid   = int(row["prompt_id"])
    if pid in done_ids:
        continue

    prompt    = str(row["prompt"])
    img_path  = IMAGE_DIR / f"prompt_{pid}.png"
    aud_path  = AUDIO_DIR / f"prompt_{pid}.wav"

    # use generated text if available, else fall back to prompt
    txt_path = TEXT_DIR / f"prompt_{pid}.txt"
    text_input = txt_path.read_text(encoding="utf-8").strip()[:200] if txt_path.exists() else prompt

    print(f"  [{pid:03d}] {prompt[:55]}...")

    try:
        inputs = {
            ModalityType.TEXT:   ib_data.load_and_transform_text([text_input], DEVICE),
            ModalityType.VISION: ib_data.load_and_transform_vision_data([str(img_path)], DEVICE),
            ModalityType.AUDIO:  _load_audio_tensor(aud_path).to(DEVICE),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        e_txt = embeddings[ModalityType.TEXT][0].cpu().numpy()
        e_img = embeddings[ModalityType.VISION][0].cpu().numpy()
        e_aud = embeddings[ModalityType.AUDIO][0].cpu().numpy()

        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        s1 = cos(e_txt, e_img)
        s2 = cos(e_txt, e_aud)
        s3 = cos(e_img, e_aud)
        tsas = (s1 + s2 + s3) / 3.0

        results.append({
            "prompt_id": pid,
            "s1_text_image": round(s1,   4),
            "s2_text_audio": round(s2,   4),
            "s3_image_audio": round(s3,  4),
            "tsas":           round(tsas, 4),
        })
        done_ids.add(pid)
        print(f"         s1={s1:.4f}  s2={s2:.4f}  s3={s3:.4f}  tsas={tsas:.4f}")

    except Exception as e:
        print(f"         ERROR: {e}")
        continue

    # save checkpoint every 10 prompts
    if len(results) % 10 == 0:
        pd.DataFrame(results).to_csv(CKPT, index=False)
        print(f"         [checkpoint saved — {len(results)} done]")

# ── save imagebind scores ─────────────────────────────────────────────────────
ib_df = pd.DataFrame(results)
ib_df.to_csv(IB_CSV, index=False)
print(f"\n  ImageBind scores saved: {IB_CSV}  ({len(ib_df)} rows)")

# ── compute WTSAS ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  Computing WTSAS")
print("=" * 65)

merged = pd.merge(df_valid, ib_df, on="prompt_id")

qt = merged["q_text"].values
qi = merged["q_image"].values
qa = merged["q_audio"].values
s1 = merged["s1_text_image"].values
s2 = merged["s2_text_audio"].values
s3 = merged["s3_image_audio"].values

# Quality-weighted coherence scores
s1_w = s1 * (W1 * qt + W2 * qi) / (W1 + W2)
s2_w = s2 * (W1 * qt + W3 * qa) / (W1 + W3)
s3_w = s3 * (W2 * qi + W3 * qa) / (W2 + W3)

wtsas     = (s1_w + s2_w + s3_w) / 3.0
wtsas_var = ((s1_w - wtsas)**2 + (s2_w - wtsas)**2 + (s3_w - wtsas)**2) / 3.0
wtsas_fin = wtsas - OPT_LAM * wtsas_var

merged["s1_weighted"]  = np.round(s1_w,      4)
merged["s2_weighted"]  = np.round(s2_w,      4)
merged["s3_weighted"]  = np.round(s3_w,      4)
merged["wtsas"]        = np.round(wtsas,     4)
merged["wtsas_final"]  = np.round(wtsas_fin, 4)

# ── correlation with judge_mean ───────────────────────────────────────────────
from scipy.stats import pearsonr

y = merged["judge_mean"].values

r_tsas,  _ = pearsonr(merged["tsas"].values,       y)
r_wtsas, _ = pearsonr(merged["wtsas_final"].values, y)
r_s1,    _ = pearsonr(s1, y)
r_s2,    _ = pearsonr(s2, y)
r_s3,    _ = pearsonr(s3, y)

print(f"\n  Results on {len(merged)} prompts (those with image + audio):")
print(f"\n  {'Metric':<35} {'Pearson r':>10}")
print(f"  {'─'*48}")
print(f"  {'s1 (text-image coherence)':<35} {r_s1:>10.4f}")
print(f"  {'s2 (text-audio coherence)':<35} {r_s2:>10.4f}")
print(f"  {'s3 (image-audio coherence)':<35} {r_s3:>10.4f}")
print(f"  {'TSAS (plain avg coherence)':<35} {r_tsas:>10.4f}")
print(f"  {'WTSAS final (quality-weighted)':<35} {r_wtsas:>10.4f}")

# ── save ─────────────────────────────────────────────────────────────────────
out_cols = ["prompt_id","prompt","q_text","q_image","q_audio",
            "judge_mean","judge_grade",
            "s1_text_image","s2_text_audio","s3_image_audio","tsas",
            "s1_weighted","s2_weighted","s3_weighted","wtsas","wtsas_final"]
merged[out_cols].to_csv(OUT_CSV, index=False)
print(f"\n  Saved: {OUT_CSV}")
print(f"\n  ALL DONE.")
