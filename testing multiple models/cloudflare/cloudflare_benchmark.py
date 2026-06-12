"""
Cloudflare Workers AI Benchmark
- Generates text, image, audio for all 50 prompts using Cloudflare
- Scores text outputs (BERTScore, ROUGE, Readability)
- Adds Cloudflare to comparison graphs alongside Mistral, Groq, Cerebras
"""

import sys
import os
import requests
import base64
import csv
import time
from pathlib import Path

BASE_DIR = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR.parent))
from prompts import get_prompts
from config import CLOUDFLARE

ACCOUNT_ID = CLOUDFLARE["account_id"]
API_TOKEN  = CLOUDFLARE["api_token"]
BASE_URL   = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run"
HEADERS    = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}

OUTPUT_TEXT  = BASE_DIR / "outputs/text"
OUTPUT_IMAGE = BASE_DIR / "outputs/images"
OUTPUT_AUDIO = BASE_DIR / "outputs/audio"
SCORES_DIR   = BASE_DIR / "outputs/scores"

for d in [OUTPUT_TEXT, OUTPUT_IMAGE, OUTPUT_AUDIO, SCORES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# GENERATION
# ============================================================================

QUOTA_EXHAUSTED = False  # set True when 429 quota hit — stop all API calls


def generate_text(prompt, prompt_id):
    global QUOTA_EXHAUSTED
    if QUOTA_EXHAUSTED:
        return None
    try:
        r = requests.post(
            f"{BASE_URL}/@cf/meta/llama-3.1-8b-instruct",
            headers=HEADERS,
            json={"messages": [{"role": "user", "content": f"Describe this scene in detail: {prompt}"}], "max_tokens": 300},
            timeout=60
        )
        if r.status_code == 200:
            text = r.json().get("result", {}).get("response", "")
            path = OUTPUT_TEXT / f"prompt_{prompt_id}.txt"
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"ORIGINAL PROMPT:\n{prompt}\n\n{'='*70}\n\nGENERATED TEXT:\n{text}")
            return text
        elif r.status_code == 429:
            QUOTA_EXHAUSTED = True
            print(f"\n  [!] Daily quota exhausted — stopping generation, will score what we have.")
            return None
        else:
            print(f"      Text error {r.status_code}: {r.text[:100]}")
            return None
    except Exception as e:
        print(f"      Text exception: {str(e)[:80]}")
        return None


def generate_image(prompt, prompt_id):
    global QUOTA_EXHAUSTED
    if QUOTA_EXHAUSTED:
        return False
    try:
        r = requests.post(
            f"{BASE_URL}/@cf/black-forest-labs/flux-1-schnell",
            headers=HEADERS,
            json={"prompt": prompt},
            timeout=120
        )
        if r.status_code == 200:
            img_b64 = r.json().get("result", {}).get("image", "")
            if img_b64:
                img_data = base64.b64decode(img_b64)
                path = OUTPUT_IMAGE / f"prompt_{prompt_id}.png"
                with open(path, "wb") as f:
                    f.write(img_data)
                return True
        elif r.status_code == 429:
            QUOTA_EXHAUSTED = True
            print(f"\n  [!] Daily quota exhausted.")
            return False
        else:
            print(f"      Image error {r.status_code}: {r.text[:100]}")
            return False
    except Exception as e:
        print(f"      Image exception: {str(e)[:80]}")
        return False


def generate_audio(text, prompt_id):
    global QUOTA_EXHAUSTED
    if QUOTA_EXHAUSTED:
        return False
    if not text:
        return False
    try:
        r = requests.post(
            f"{BASE_URL}/@cf/myshell-ai/melotts",
            headers=HEADERS,
            json={"prompt": text},
            timeout=60
        )
        if r.status_code == 200:
            audio_b64 = r.json().get("result", {}).get("audio", "")
            if audio_b64:
                audio_data = base64.b64decode(audio_b64)
                path = OUTPUT_AUDIO / f"prompt_{prompt_id}.wav"
                with open(path, "wb") as f:
                    f.write(audio_data)
                return True
        elif r.status_code == 429:
            QUOTA_EXHAUSTED = True
            print(f"\n  [!] Daily quota exhausted.")
            return False
        else:
            print(f"      Audio error {r.status_code}: {r.text[:100]}")
            return False
    except Exception as e:
        print(f"      Audio exception: {str(e)[:80]}")
        return False


# ============================================================================
# SCORING
# ============================================================================

def compute_bertscore(reference, candidate):
    try:
        from bert_score import score
        P, R, F1 = score([candidate], [reference], lang="en", verbose=False)
        return round(F1.item(), 4)
    except Exception as e:
        print(f"      BERTScore error: {e}")
        return 0.0


def compute_rouge(reference, candidate):
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return round(scores["rougeL"].fmeasure, 4)
    except Exception as e:
        print(f"      ROUGE error: {e}")
        return 0.0


def compute_readability(text):
    try:
        from textstat import flesch_kincaid_grade
        score = flesch_kincaid_grade(text)
        return round(min(10.0, max(0.0, score / 10.0)), 4)
    except Exception as e:
        print(f"      Readability error: {e}")
        return 0.0


def score_cloudflare_text(prompts):
    print("\n  Scoring Cloudflare text outputs...")
    results = []

    for i, prompt in enumerate(prompts, 1):
        path = OUTPUT_TEXT / f"prompt_{i}.txt"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract generated text (after the separator)
        parts = content.split("GENERATED TEXT:\n")
        text = parts[1].strip() if len(parts) > 1 else content

        bertscore   = compute_bertscore(prompt, text)
        rouge       = compute_rouge(prompt, text)
        readability = compute_readability(text)

        results.append({
            "prompt_id":   i,
            "model":       "cloudflare",
            "bertscore":   bertscore,
            "rouge":       rouge,
            "readability": readability,
        })

    return results


def save_csv(results, filename, fieldnames):
    """Save scores to CSV, replacing any existing cloudflare rows."""
    csv_path = SCORES_DIR / filename
    existing = []
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            existing = [row for row in csv.DictReader(f) if row["model"] != "cloudflare"]
    all_rows = existing + results
    if all_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"  Saved: {csv_path}")


# ── Image scoring ─────────────────────────────────────────────────────────────

def score_cloudflare_images(prompts):
    print("\n  Scoring Cloudflare image outputs...")
    try:
        import torch
        from PIL import Image as PILImage
        from transformers import CLIPProcessor, CLIPModel
        import requests as req
        import torch.nn as nn
    except ImportError as e:
        print(f"  Skipping image scoring — missing package: {e}")
        return []

    # CLIP
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()

    # LAION Aesthetic predictor (linear MLP on CLIP features)
    class AestheticMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(768, 1024), nn.Dropout(0.2),
                nn.Linear(1024, 128), nn.Dropout(0.2),
                nn.Linear(128, 64),   nn.Dropout(0.1),
                nn.Linear(64, 16),    nn.Linear(16, 1))
        def forward(self, x): return self.layers(x)

    weights_path = BASE_DIR / "aesthetic_weights.pth"
    if not weights_path.exists():
        print("  Downloading aesthetic predictor weights (~17MB)...")
        url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
        weights_path.write_bytes(req.get(url, timeout=60).content)
    aes_model = AestheticMLP()
    aes_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    aes_model.eval()

    results = []
    images  = sorted(OUTPUT_IMAGE.glob("prompt_*.png"),
                     key=lambda p: int(p.stem.split("_")[1]))
    for img_path in images:
        pid    = int(img_path.stem.split("_")[1])
        prompt = prompts[pid - 1]
        try:
            image  = PILImage.open(img_path).convert("RGB")
            inputs = clip_proc(text=[prompt], images=image, return_tensors="pt",
                               padding=True, truncation=True, max_length=77)
            with torch.no_grad():
                img_feat = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
                txt_feat = clip_model.get_text_features(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                img_feat_n = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat_n = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                clip_score = round(max(0.0, (img_feat_n * txt_feat_n).sum().item()), 4)
                aes_score  = round(min(10.0, max(0.0, aes_model(img_feat_n).item())), 4)
        except Exception as e:
            print(f"    prompt_{pid} error: {e}")
            clip_score, aes_score = 0.0, 0.0

        results.append({"prompt_id": pid, "model": "cloudflare",
                         "clip": clip_score, "aesthetic": aes_score})
        print(f"    prompt_{pid:02d}: CLIP={clip_score:.4f}  Aesthetic={aes_score:.4f}")

    return results


# ── Audio scoring ─────────────────────────────────────────────────────────────

def score_cloudflare_audio(prompts):
    print("\n  Scoring Cloudflare audio outputs...")

    # Check packages
    try:
        import torch
        from transformers import ClapModel, ClapProcessor
        import soundfile as sf
        CLAP_OK = True
    except ImportError as e:
        CLAP_OK = False
        print(f"  CLAP unavailable: {e}")

    try:
        import whisper
        from jiwer import wer as jiwer_wer
        import soundfile as sf_wer
        WHISPER_OK = True
        wmodel = whisper.load_model("base")
    except ImportError as e:
        WHISPER_OK = False
        print(f"  WER unavailable: {e}  →  pip install openai-whisper jiwer soundfile")

    try:
        from speechmos import dnsmos as dnsmos_module
        import soundfile as sf_mos
        MOS_OK = True
    except ImportError as e:
        MOS_OK = False
        print(f"  MOS unavailable: {e}  →  pip install speechmos onnxruntime soundfile")

    if not any([CLAP_OK, WHISPER_OK, MOS_OK]):
        print("  Skipping audio scoring — no packages available.")
        return []

    if CLAP_OK:
        clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        clap_proc  = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        clap_model.eval()

    results = []
    audios  = sorted(OUTPUT_AUDIO.glob("prompt_*.wav"),
                     key=lambda p: int(p.stem.split("_")[1]))

    for audio_path in audios:
        pid    = int(audio_path.stem.split("_")[1])
        prompt = prompts[pid - 1]

        # CLAP — model requires 48000 Hz, resample if needed
        clap_score = 0.0
        if CLAP_OK:
            try:
                audio, sr = sf.read(str(audio_path))
                if audio.ndim > 1: audio = audio.mean(axis=1)
                if sr != 48000:
                    from scipy import signal as _sig
                    n48 = int(len(audio) * 48000 / sr)
                    audio = _sig.resample(audio, n48)
                inp = clap_proc(text=[prompt], audios=[audio], return_tensors="pt",
                                padding=True, sampling_rate=48000)
                with torch.no_grad():
                    af = clap_model.get_audio_features(input_features=inp["input_features"])
                    tf = clap_model.get_text_features(
                        input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
                af = af / af.norm(dim=-1, keepdim=True)
                tf = tf / tf.norm(dim=-1, keepdim=True)
                clap_score = round(max(0.0, (af * tf).sum().item()), 4)
            except Exception as e:
                print(f"    CLAP error prompt_{pid}: {e}")

        # WER — load with soundfile to avoid ffmpeg dependency
        wer_score = 0.0
        if WHISPER_OK:
            try:
                import numpy as np
                audio_wer, orig_sr = sf_wer.read(str(audio_path), dtype='float32')
                if audio_wer.ndim > 1:
                    audio_wer = audio_wer.mean(axis=1)
                if orig_sr != 16000:
                    from scipy import signal as scipy_signal
                    n_samples = int(len(audio_wer) * 16000 / orig_sr)
                    audio_wer = scipy_signal.resample(audio_wer, n_samples).astype('float32')
                result     = wmodel.transcribe(audio_wer)
                hypothesis = result["text"].strip().lower()
                wer_score  = round(min(1.0, max(0.0, jiwer_wer(prompt.lower(), hypothesis))), 4)
            except Exception as e:
                print(f"    WER error prompt_{pid}: {e}")

        # MOS — correct speechmos API: dnsmos.run(audio_array, 16000)
        mos_score = 0.0
        if MOS_OK:
            try:
                import numpy as np
                from scipy import signal as scipy_signal
                audio_mos, sr_mos = sf_mos.read(str(audio_path), dtype='float32')
                if audio_mos.ndim > 1:
                    audio_mos = audio_mos.mean(axis=1)
                if sr_mos != 16000:
                    n_samples = int(len(audio_mos) * 16000 / sr_mos)
                    audio_mos = scipy_signal.resample(audio_mos, n_samples).astype('float32')
                r = dnsmos_module.run(audio_mos, 16000, model_type='dnsmos',
                                      return_df=True, verbose=False)
                # result is a dict: {'ovrl_mos': ..., 'sig_mos': ..., 'bak_mos': ..., 'p808_mos': ...}
                if hasattr(r, 'iloc'):
                    row = r.iloc[0].to_dict()
                else:
                    row = r
                mos_val = float(row.get('ovrl_mos', row.get('OVRL', row.get('mos', 0))))
                mos_score = round(min(5.0, max(1.0, mos_val)), 4)
            except Exception as e:
                print(f"    MOS error prompt_{pid}: {e}")

        results.append({"prompt_id": pid, "model": "cloudflare",
                         "clap": clap_score, "mos": mos_score, "wer": wer_score})
        print(f"    prompt_{pid:02d}: CLAP={clap_score:.4f}  MOS={mos_score:.4f}  WER={wer_score:.4f}")

    return results


# ============================================================================
# GRAPHING
# ============================================================================

def regenerate_graphs():
    """Regenerate text comparison graphs including Cloudflare"""
    import csv as csv_mod
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    sys.path.insert(0, str(BASE_DIR.parent))
    from config import BIG_PLAYERS_BENCHMARKS

    graphs_dir = BASE_DIR / "outputs/graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    scores_path = SCORES_DIR / "text_scores.csv"

    if not scores_path.exists():
        print("  No text_scores.csv found - skipping graphs")
        return

    # Load scores
    scores = {}
    with open(scores_path, "r", encoding="utf-8") as f:
        for row in csv_mod.DictReader(f):
            m = row["model"]
            if m not in scores:
                scores[m] = []
            scores[m].append(row)

    # Calculate averages
    def avg(scores, model, metric):
        vals = [float(r[metric]) for r in scores.get(model, []) if metric in r]
        return round(sum(vals) / len(vals), 4) if vals else 0

    small_models  = [m for m in scores if m not in BIG_PLAYERS_BENCHMARKS]
    big_players   = list(BIG_PLAYERS_BENCHMARKS.keys())

    COLORS = {
        "mistral":    "#7B68EE",
        "groq":       "#20B2AA",
        "cerebras":   "#FFD700",
        "deepseek":   "#FF6347",
        "cloudflare": "#FF8C00",
    }
    BIG_COLORS = {"claude": "#1E90FF", "chatgpt": "#00CED1", "gemini": "#DC143C"}

    def make_bar_chart(metric, ylabel, title, filename):
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#f8f9fa")
        ax.grid(axis="y", alpha=0.4, linestyle="--", color="gray")

        x = np.arange(len(small_models))
        bar_colors = [COLORS.get(m, "steelblue") for m in small_models]
        bars = ax.bar(x, [avg(scores, m, metric) for m in small_models],
                      color=bar_colors, alpha=0.9, width=0.5, label="Small models",
                      edgecolor="white", linewidth=0.8)

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        # Big players as dashed lines
        line_styles = ["--", "-.", ":"]
        for i, (player, color) in enumerate(BIG_COLORS.items()):
            val = BIG_PLAYERS_BENCHMARKS[player].get(metric, 0)
            ax.axhline(val, color=color, linestyle=line_styles[i % 3],
                       linewidth=2, label=f"{player.capitalize()} = {val}", alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(small_models, fontsize=12)
        ax.set_xlabel("Model", fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
        ax.legend(fontsize=10, loc="upper right")
        ax.spines[["top","right"]].set_visible(False)

        plt.tight_layout()
        plt.savefig(graphs_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {filename}")

    print("\n  Regenerating text graphs...")
    make_bar_chart("bertscore",   "BERTScore (0-1)",  "BERTScore — Small Models vs Big Players",   "bertscore_comparison.png")
    make_bar_chart("rouge",       "ROUGE (0-1)",       "ROUGE Score — Small Models vs Big Players",  "rouge_comparison.png")
    make_bar_chart("readability", "Score (0-10)",      "Readability — Small Models vs Big Players",  "readability_comparison.png")

    # Line graph (all metrics)
    metrics = ["bertscore", "rouge", "readability"]
    xlabels = ["BERTScore", "ROUGE", "Readability\n(0-10)"]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")
    ax.grid(alpha=0.4, linestyle="--")

    for m in small_models:
        vals = [avg(scores, m, metric) for metric in metrics]
        ax.plot(xlabels, vals, marker="o", label=m.capitalize(),
                color=COLORS.get(m, "steelblue"), linewidth=2)
        for xi, yi in zip(xlabels, vals):
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8,
                        color=COLORS.get(m, "steelblue"))

    for i, (player, color) in enumerate(BIG_COLORS.items()):
        vals = [BIG_PLAYERS_BENCHMARKS[player].get(metric, 0) for metric in metrics]
        ax.plot(xlabels, vals, marker="D", linestyle="--",
                label=f"{player.capitalize()} (Big Player)",
                color=color, linewidth=2)

    ax.set_title("Text Model Performance — Small Models vs Big Players", fontsize=15, fontweight="bold")
    ax.set_ylabel("Score", fontsize=13)
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(graphs_dir / "text_line_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: text_line_comparison.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    prompts = get_prompts()
    total = len(prompts)
    generated_texts = {}

    print("=" * 65)
    print("  CLOUDFLARE WORKERS AI — FULL BENCHMARK")
    print(f"  Running {total} prompts across Text + Image + Audio")
    print("=" * 65)

    # Generation
    for i, prompt in enumerate(prompts, 1):
        if QUOTA_EXHAUSTED:
            print(f"\n  Quota exhausted at prompt {i-1}. Skipping remaining generation.")
            break

        print(f"\nPrompt {i}/{total}: {prompt[:55]}...")

        print("    Text...", end="", flush=True)
        if (OUTPUT_TEXT / f"prompt_{i}.txt").exists():
            with open(OUTPUT_TEXT / f"prompt_{i}.txt", encoding="utf-8") as f:
                content = f.read()
            parts = content.split("GENERATED TEXT:\n")
            text = parts[1].strip() if len(parts) > 1 else content
            generated_texts[i] = text
            print(" ✓ (cached)")
        else:
            text = generate_text(prompt, i)
            generated_texts[i] = text
            print(" ✓" if text else " ✗")
            if QUOTA_EXHAUSTED:
                break
            time.sleep(0.5)

        print("    Image...", end="", flush=True)
        if (OUTPUT_IMAGE / f"prompt_{i}.png").exists():
            print(" ✓ (cached)")
        else:
            ok = generate_image(prompt, i)
            print(" ✓" if ok else " ✗")
            if QUOTA_EXHAUSTED:
                break
            time.sleep(0.5)

        text_for_audio = text if text else prompt
        print("    Audio...", end="", flush=True)
        if (OUTPUT_AUDIO / f"prompt_{i}.wav").exists():
            print(" ✓ (cached)")
        else:
            ok = generate_audio(text_for_audio[:300], i)
            print(" ✓" if ok else " ✗")
            time.sleep(0.5)

    # ── Text scoring
    print("\n" + "=" * 65)
    print("  SCORING — TEXT")
    print("=" * 65)
    text_results = score_cloudflare_text(prompts)
    save_csv(text_results, "text_scores.csv",
             ["prompt_id", "model", "bertscore", "rouge", "readability"])
    if text_results:
        print(f"  Avg BERTScore:   {sum(r['bertscore']   for r in text_results)/len(text_results):.4f}")
        print(f"  Avg ROUGE:       {sum(r['rouge']       for r in text_results)/len(text_results):.4f}")
        print(f"  Avg Readability: {sum(r['readability'] for r in text_results)/len(text_results):.4f}")

    # ── Image scoring
    print("\n" + "=" * 65)
    print("  SCORING — IMAGE")
    print("=" * 65)
    image_results = score_cloudflare_images(prompts)
    if image_results:
        save_csv(image_results, "image_scores.csv",
                 ["prompt_id", "model", "clip", "aesthetic"])
        print(f"  Avg CLIP:      {sum(r['clip']      for r in image_results)/len(image_results):.4f}")
        print(f"  Avg Aesthetic: {sum(r['aesthetic'] for r in image_results)/len(image_results):.4f}")

    # ── Audio scoring
    print("\n" + "=" * 65)
    print("  SCORING — AUDIO")
    print("=" * 65)
    audio_results = score_cloudflare_audio(prompts)
    if audio_results:
        save_csv(audio_results, "audio_scores.csv",
                 ["prompt_id", "model", "clap", "mos", "wer"])
        print(f"  Avg CLAP: {sum(r['clap'] for r in audio_results)/len(audio_results):.4f}")
        print(f"  Avg MOS:  {sum(r['mos']  for r in audio_results)/len(audio_results):.4f}")
        print(f"  Avg WER:  {sum(r['wer']  for r in audio_results)/len(audio_results):.4f}")

    # ── Graphs
    print("\n" + "=" * 65)
    print("  REGENERATING GRAPHS")
    print("=" * 65)
    regenerate_graphs()

    print("\n" + "=" * 65)
    print("  DONE")
    print("  Text   → cloudflare/outputs/text/")
    print("  Image  → cloudflare/outputs/images/")
    print("  Audio  → cloudflare/outputs/audio/")
    print("  Scores → cloudflare/outputs/scores/")
    print("  Graphs → cloudflare/outputs/graphs/")
    print("=" * 65)


if __name__ == "__main__":
    main()
