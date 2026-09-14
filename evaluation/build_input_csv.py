"""
Build the input CSV for full_pipeline.py from existing generated outputs.

Scans outputs/text/, outputs/images/, outputs/audio/ and matches files
to prompts. Works for both the 53-prompt set and 500-prompt set.

Usage:
    python build_input_csv.py

Outputs: prompts_with_outputs.csv  (in same folder as this script)
"""

import sys
from pathlib import Path
import pandas as pd

# -- paths (relative to this script) ----------------------------------------
BASE      = Path(__file__).parent.parent          # cloudflare/
TEXT_DIR  = BASE / "outputs" / "text"
IMAGE_DIR = BASE / "outputs" / "images"
AUDIO_DIR = BASE / "outputs" / "audio"
PROMPTS_CSV = Path(__file__).parent / "prompts_500.csv"  # your 500 prompts
OUT_CSV     = Path(__file__).parent / "prompts_with_outputs.csv"

# -- load prompts ------------------------------------------------------------
df_prompts = pd.read_csv(PROMPTS_CSV)
print(f"Loaded {len(df_prompts)} prompts from {PROMPTS_CSV.name}")
print(f"Scanning outputs in: {BASE / 'outputs'}\n")

# -- match files -------------------------------------------------------------
rows = []
missing = []

for _, row in df_prompts.iterrows():
    pid   = int(row["prompt_id"])
    pmpt  = str(row["prompt"])

    # look for text file
    txt_path = TEXT_DIR / f"prompt_{pid}.txt"
    # look for image (try .png then .jpg)
    img_path = IMAGE_DIR / f"prompt_{pid}.png"
    if not img_path.exists():
        img_path = IMAGE_DIR / f"prompt_{pid}.jpg"
    # look for audio (try .wav then .mp3)
    aud_path = AUDIO_DIR / f"prompt_{pid}.wav"
    if not aud_path.exists():
        aud_path = AUDIO_DIR / f"prompt_{pid}.mp3"

    has_text  = txt_path.exists()
    has_image = img_path.exists()
    has_audio = aud_path.exists()

    if has_text and has_image and has_audio:
        generated_text = txt_path.read_text(encoding="utf-8").strip()
        rows.append({
            "prompt_id":      pid,
            "prompt":         pmpt,
            "generated_text": generated_text,
            "image_path":     str(img_path),
            "audio_path":     str(aud_path),
        })
    else:
        missing.append({
            "prompt_id": pid,
            "has_text":  has_text,
            "has_image": has_image,
            "has_audio": has_audio,
        })

# -- report ------------------------------------------------------------------
print(f"  Found complete outputs : {len(rows)}")
print(f"  Missing outputs        : {len(missing)}")

if missing:
    print(f"\n  First 10 missing:")
    for m in missing[:10]:
        flags = []
        if not m["has_text"]:  flags.append("text")
        if not m["has_image"]: flags.append("image")
        if not m["has_audio"]: flags.append("audio")
        print(f"    prompt_{m['prompt_id']:03d} — missing: {', '.join(flags)}")

if not rows:
    print("\n  No complete outputs found.")
    print("  Run your Cloudflare generation pipeline first to create:")
    print(f"    {TEXT_DIR}/prompt_N.txt")
    print(f"    {IMAGE_DIR}/prompt_N.png")
    print(f"    {AUDIO_DIR}/prompt_N.wav")
    sys.exit(0)

# -- save --------------------------------------------------------------------
df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_CSV, index=False)
print(f"\n  Saved {len(df_out)} rows to: {OUT_CSV}")
print(f"\n  Next step:")
print(f"    python full_pipeline.py")
