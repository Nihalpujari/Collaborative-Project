"""
Audio scoring for Cloudflare: CLAP + MOS + WER for all 56 prompts.
Run from the cloudflare/ folder:
    python score_audio_only.py
"""
import sys, os, csv
from pathlib import Path

BASE_DIR = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR.parent))

from prompts import get_prompts
prompts      = get_prompts()
OUTPUT_AUDIO = BASE_DIR / "outputs/audio"
SCORES_DIR   = BASE_DIR / "outputs/scores"

import soundfile as sf
import numpy as np
from scipy import signal as scipy_signal

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading Whisper base...")
import whisper
from jiwer import wer as jiwer_wer
wmodel = whisper.load_model("base")

print("Loading CLAP...")
import torch
from transformers import ClapModel, ClapProcessor
clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
clap_proc  = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
clap_model.eval()

print("Loading speechmos...")
from speechmos import dnsmos as dnsmos_module

print("All models ready. Scoring 56 audio files...\n")

# ── Score each file ───────────────────────────────────────────────────────────
results = []
audios  = sorted(OUTPUT_AUDIO.glob("prompt_*.wav"),
                 key=lambda p: int(p.stem.split("_")[1]))

for audio_path in audios:
    pid    = int(audio_path.stem.split("_")[1])
    prompt = prompts[pid - 1]

    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # resample helpers
    def to_sr(a, target):
        if sr == target:
            return a
        return scipy_signal.resample(a, int(len(a) * target / sr)).astype("float32")

    # CLAP — needs 48000 Hz
    clap_score = 0.0
    try:
        a48 = to_sr(audio, 48000)
        inp = clap_proc(text=[prompt], audios=[a48], return_tensors="pt",
                        padding=True, sampling_rate=48000)
        with torch.no_grad():
            af = clap_model.get_audio_features(input_features=inp["input_features"])
            tf = clap_model.get_text_features(
                input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
        af = af / af.norm(dim=-1, keepdim=True)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        clap_score = round(max(0.0, (af * tf).sum().item()), 4)
    except Exception as e:
        print(f"  CLAP error prompt_{pid}: {e}")

    # MOS — needs 16000 Hz
    mos_score = 0.0
    try:
        a16 = to_sr(audio, 16000)
        r = dnsmos_module.run(a16, 16000, model_type="dnsmos",
                              return_df=True, verbose=False)
        mos_score = round(min(5.0, max(1.0, float(r.get("ovrl_mos", 0)))), 4)
    except Exception as e:
        print(f"  MOS error prompt_{pid}: {e}")

    # WER — needs 16000 Hz numpy array (no ffmpeg)
    wer_score = 0.0
    try:
        a16 = to_sr(audio, 16000)
        hyp       = wmodel.transcribe(a16)["text"].strip().lower()
        wer_score = round(min(1.0, max(0.0, jiwer_wer(prompt.lower(), hyp))), 4)
    except Exception as e:
        print(f"  WER error prompt_{pid}: {e}")

    results.append({"prompt_id": pid, "model": "cloudflare",
                    "clap": clap_score, "mos": mos_score, "wer": wer_score})
    print(f"  prompt_{pid:02d}: CLAP={clap_score:.4f}  MOS={mos_score:.4f}  WER={wer_score:.4f}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
out_path = SCORES_DIR / "audio_scores.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["prompt_id", "model", "clap", "mos", "wer"])
    w.writeheader()
    w.writerows(results)

avg_clap = sum(r["clap"] for r in results) / len(results)
avg_mos  = sum(r["mos"]  for r in results) / len(results)
avg_wer  = sum(r["wer"]  for r in results) / len(results)
print(f"\nSaved {len(results)} rows -> {out_path}")
print(f"Avg CLAP={avg_clap:.4f}  MOS={avg_mos:.4f}  WER={avg_wer:.4f}")
print("\nDone! Now run: python cf_vs_bigplayers.py")
