"""
CLIP Score Evaluation — Image-Text Alignment (Gourav)
======================================================
Standalone, validated CLIP scorer for the Cloudflare image benchmark.

Improvements over score_cloudflare_images() in cloudflare_benchmark.py:
  1. LOUD failures  — a broken image/prompt raises or is reported, never a silent 0.0
  2. Validation mode — mismatched-pair sanity test proves CLIP is measuring alignment
  3. Missing-image report — tells you exactly which prompts still need generation
  4. Both metric conventions — raw cosine AND standard CLIPScore (2.5 x max(cos,0),
     Hessel et al. 2021) so comparisons against published numbers are apples-to-apples
  5. GPU support + batching — ~10x faster on CUDA
  6. Preserves non-cloudflare rows when rewriting the CSV

Usage:
    python score_image_clip.py              # score all images, write CSV
    python score_image_clip.py --validate   # run mismatched-pair sanity test only
    python score_image_clip.py --all        # score + validate

Requires: torch, transformers, Pillow
"""

import sys
import csv
import argparse
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR.parent))

from prompts import get_prompts

IMAGE_DIR  = BASE_DIR / "outputs/images"
SCORES_DIR = BASE_DIR / "outputs/scores"
CSV_PATH   = SCORES_DIR / "image_scores.csv"

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
BATCH_SIZE    = 8


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_clip():
    import torch
    from transformers import CLIPProcessor, CLIPModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[clip] loading {CLIP_MODEL_ID} on {device} ...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    return model, proc, device


# ─────────────────────────────────────────────────────────────────────────────
# Core scoring
# ─────────────────────────────────────────────────────────────────────────────

def clip_cosine(model, proc, device, images, texts):
    """Return list of cosine similarities for (image, text) pairs. Raises on failure."""
    import torch

    # Warn if any prompt exceeds CLIP's 77-token context (it gets truncated)
    tok = proc.tokenizer(texts)["input_ids"]
    for t, ids in zip(texts, tok):
        if len(ids) > 77:
            print(f"  [warn] prompt truncated to 77 tokens: '{t[:60]}...'")

    inputs = proc(text=texts, images=images, return_tensors="pt",
                  padding=True, truncation=True, max_length=77).to(device)
    with torch.no_grad():
        img_f = model.get_image_features(pixel_values=inputs["pixel_values"])
        txt_f = model.get_text_features(input_ids=inputs["input_ids"],
                                        attention_mask=inputs["attention_mask"])
    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
    txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
    return (img_f * txt_f).sum(dim=-1).cpu().tolist()


def collect_pairs(prompts):
    """Return (found, missing) — found is list of (pid, image_path, prompt)."""
    found, missing = [], []
    for pid in range(1, len(prompts) + 1):
        p = IMAGE_DIR / f"prompt_{pid}.png"
        if p.exists() and p.stat().st_size > 0:
            found.append((pid, p, prompts[pid - 1]))
        else:
            missing.append(pid)
    return found, missing


def score_all(prompts):
    from PIL import Image as PILImage

    model, proc, device = load_clip()
    pairs, missing = collect_pairs(prompts)

    if missing:
        print(f"\n[!] {len(missing)} images MISSING — regenerate these before final results:")
        for pid in missing:
            print(f"      prompt_{pid}: {prompts[pid-1][:70]}")
        print()

    results, failures = [], []
    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i:i + BATCH_SIZE]
        try:
            imgs  = [PILImage.open(p).convert("RGB") for _, p, _ in batch]
            texts = [t for _, _, t in batch]
            cosines = clip_cosine(model, proc, device, imgs, texts)
        except Exception as e:
            # Batch failed → retry one-by-one so a single bad file can't sink 8 scores
            print(f"  [warn] batch failed ({e}); retrying individually")
            cosines = []
            for pid, p, t in batch:
                try:
                    img = PILImage.open(p).convert("RGB")
                    cosines.append(clip_cosine(model, proc, device, [img], [t])[0])
                except Exception as e2:
                    failures.append((pid, str(e2)))
                    cosines.append(None)

        for (pid, _, _), cos in zip(batch, cosines):
            if cos is None:
                continue
            results.append({
                "prompt_id":  pid,
                "model":      "cloudflare",
                "clip":       round(max(0.0, cos), 4),                  # raw cosine
                "clipscore":  round(2.5 * max(0.0, cos), 4),            # Hessel et al. 2021
            })
            print(f"  prompt_{pid:02d}: cos={cos:.4f}  CLIPScore={2.5*max(0,cos):.4f}")

    if failures:
        print(f"\n[!] {len(failures)} images FAILED to score (excluded from CSV, NOT zero-filled):")
        for pid, err in failures:
            print(f"      prompt_{pid}: {err[:80]}")

    return results, missing, failures


# ─────────────────────────────────────────────────────────────────────────────
# Validation — mismatched-pair sanity test
# ─────────────────────────────────────────────────────────────────────────────

def validate(prompts, n_samples=15, seed=42):
    """
    Proof that CLIP is measuring alignment: each image scored against its OWN
    prompt must beat the same image scored against a RANDOM WRONG prompt.
    Passes if matched > mismatched for >= 90% of sampled images and the mean
    gap is > 0.05 cosine.
    """
    from PIL import Image as PILImage

    model, proc, device = load_clip()
    pairs, _ = collect_pairs(prompts)
    rng = random.Random(seed)
    sample = rng.sample(pairs, min(n_samples, len(pairs)))

    wins, gaps = 0, []
    print(f"\n[validate] mismatched-pair test on {len(sample)} images")
    print(f"{'prompt':>9} | {'matched':>8} | {'mismatch':>8} | verdict")
    print("-" * 46)
    for pid, path, own_prompt in sample:
        wrong = rng.choice([t for q, _, t in pairs if q != pid])
        img = PILImage.open(path).convert("RGB")
        m, w = clip_cosine(model, proc, device, [img, img], [own_prompt, wrong])
        gap = m - w
        gaps.append(gap)
        ok = m > w
        wins += ok
        print(f"prompt_{pid:02d} | {m:8.4f} | {w:8.4f} | {'PASS' if ok else 'FAIL'}")

    win_rate = wins / len(sample)
    mean_gap = sum(gaps) / len(gaps)
    print("-" * 46)
    print(f"win rate: {win_rate:.0%}   mean gap: {mean_gap:+.4f}")
    passed = win_rate >= 0.90 and mean_gap > 0.05
    print(f"VALIDATION {'PASSED — CLIP is measuring alignment' if passed else 'FAILED — investigate pipeline'}")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# CSV output (preserves rows from other models)
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(results):
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    existing = []
    if CSV_PATH.exists():
        with open(CSV_PATH, encoding="utf-8") as f:
            existing = [r for r in csv.DictReader(f) if r.get("model") != "cloudflare"]

    # Keep aesthetic column if it exists in old cloudflare rows
    old_aes = {}
    if CSV_PATH.exists():
        with open(CSV_PATH, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("model") == "cloudflare" and "aesthetic" in r:
                    old_aes[r["prompt_id"]] = r["aesthetic"]
    for r in results:
        r["aesthetic"] = old_aes.get(str(r["prompt_id"]), "")

    fieldnames = ["prompt_id", "model", "clip", "clipscore", "aesthetic"]
    rows = existing + results
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] wrote {len(rows)} rows -> {CSV_PATH}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="run sanity test only")
    ap.add_argument("--all", action="store_true", help="score + validate")
    args = ap.parse_args()

    prompts = get_prompts()

    if args.validate:
        ok = validate(prompts)
        sys.exit(0 if ok else 1)

    results, missing, failures = score_all(prompts)
    if results:
        save_csv(results)
        clips = [r["clip"] for r in results]
        print(f"\nSummary: n={len(results)}  "
              f"cos avg={sum(clips)/len(clips):.4f}  "
              f"min={min(clips):.4f}  max={max(clips):.4f}  "
              f"missing={len(missing)}  failed={len(failures)}")

    if args.all:
        validate(prompts)