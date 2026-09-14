"""
Cloudflare vs Big Players (Claude, ChatGPT, Gemini)
Generates comparison graphs for Text, Image, and Audio metrics.

Run after all scoring scripts are complete:
  python score_cf_images.py   (image scores)
  python score_cf_audio.py    (audio scores)
  python cf_vs_bigplayers.py  (this file — graphs)
"""

import os
import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
from pathlib import Path

BASE       = Path(os.path.abspath(__file__)).parent
SCORES_DIR = BASE / "outputs/scores"
GRAPHS_DIR = BASE / "outputs/graphs"

GRAPHS_TEXT  = GRAPHS_DIR / "text"
GRAPHS_IMAGE = GRAPHS_DIR / "image"
GRAPHS_AUDIO = GRAPHS_DIR / "audio"
for d in [GRAPHS_TEXT, GRAPHS_IMAGE, GRAPHS_AUDIO]:
    d.mkdir(parents=True, exist_ok=True)

# ── Big player benchmarks ─────────────────────────────────────────────────────
BIG_PLAYERS = {
    "Claude":  {
        "bertscore": 0.851, "rouge": 0.078, "readability": 1.85,
        "clip": 0.312, "aesthetic": 7.20,
        "clap": 0.74,  "mos": 4.10, "wer": 0.072,
    },
    "ChatGPT": {
        "bertscore": 0.832, "rouge": 0.071, "readability": 1.72,
        "clip": 0.305, "aesthetic": 7.05,
        "clap": 0.72,  "mos": 4.05, "wer": 0.085,
    },
    "Gemini":  {
        "bertscore": 0.796, "rouge": 0.068, "readability": 1.68,
        "clip": 0.298, "aesthetic": 6.90,
        "clap": 0.76,  "mos": 4.20, "wer": 0.068,
    },
}

COLORS = {
    "Cloudflare": "#FF8C00",
    "Claude":     "#1E90FF",
    "ChatGPT":    "#00CED1",
    "Gemini":     "#DC143C",
}


# ── Load scores ───────────────────────────────────────────────────────────────
def load_cf_avg(csv_file, keys):
    path = SCORES_DIR / csv_file
    if not path.exists():
        return None
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if r["model"] == "cloudflare"]
    if not rows:
        return None
    def avg(key):
        vals = [float(r[key]) for r in rows if float(r[key]) > 0]
        return round(sum(vals) / len(vals), 4) if vals else 0.0
    return {k: avg(k) for k in keys}

text_cf  = load_cf_avg("text_scores.csv",  ["bertscore", "rouge", "readability"])
image_cf = load_cf_avg("image_scores.csv", ["clip", "aesthetic"])
audio_cf = load_cf_avg("audio_scores.csv", ["clap", "mos", "wer"])


# ── Shared helpers ────────────────────────────────────────────────────────────
def bar_chart(title, metric, ylabel, cf_val, filename, lower_better=False):
    vals   = [cf_val] + [BIG_PLAYERS[m][metric] for m in BIG_PLAYERS]
    models = ["Cloudflare"] + list(BIG_PLAYERS.keys())
    colors = [COLORS[m] for m in models]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")
    ax.grid(axis="y", alpha=0.4, linestyle="--")

    bars = ax.bar(models, vals, color=colors, alpha=0.9, width=0.5, edgecolor="white")
    bars[0].set_edgecolor("black"); bars[0].set_linewidth(1.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + max(vals) * 0.01,
                f"{h:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    note = " (lower is better)" if lower_better else ""
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_ylabel(ylabel + note, fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename.name}")


def grouped_bar(title, metric_labels, metric_keys, cf_scores, filename, lower_better_keys=None):
    lower_better_keys = lower_better_keys or []
    models     = ["Cloudflare"] + list(BIG_PLAYERS.keys())
    all_scores = [cf_scores] + [BIG_PLAYERS[m] for m in BIG_PLAYERS]
    x     = np.arange(len(metric_labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")
    ax.grid(axis="y", alpha=0.4, linestyle="--")

    for i, (model, scores) in enumerate(zip(models, all_scores)):
        offset = (i - len(models) / 2 + 0.5) * width
        vals   = [scores[k] for k in metric_keys]
        bars   = ax.bar(x + offset, vals, width, label=model,
                        color=COLORS[model], alpha=0.9, edgecolor="white")
        if model == "Cloudflare":
            for bar in bars:
                bar.set_edgecolor("black"); bar.set_linewidth(1.5)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename.name}")


def line_chart(title, metric_labels, metric_keys, cf_scores, filename):
    cf_vals = [cf_scores[k] for k in metric_keys]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")
    ax.grid(alpha=0.4, linestyle="--")

    ax.plot(metric_labels, cf_vals, marker="o", color=COLORS["Cloudflare"],
            linewidth=2.5, label="Cloudflare", zorder=5)
    ax.fill_between(metric_labels, cf_vals, alpha=0.12, color=COLORS["Cloudflare"])
    for xi, yi in zip(metric_labels, cf_vals):
        ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10, fontweight="bold",
                    color=COLORS["Cloudflare"])

    line_styles = ["--", "-.", ":"]; markers = ["D", "s", "*"]
    for i, (player, bp) in enumerate(BIG_PLAYERS.items()):
        vals = [bp[k] for k in metric_keys]
        ax.plot(metric_labels, vals, marker=markers[i], linestyle=line_styles[i],
                color=COLORS[player], linewidth=2, label=f"{player} (Big Player)", alpha=0.9)
        for xi, yi in zip(metric_labels, vals):
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9, color=COLORS[player])

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("Score", fontsize=13)
    ax.legend(fontsize=10, loc="upper left", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename.name}")


# ── TEXT graphs ───────────────────────────────────────────────────────────────
if text_cf:
    print(f"\nText scores loaded: {text_cf}")
    print("Generating TEXT graphs...")
    bar_chart("BERTScore: Semantic Similarity",      "bertscore",   "BERTScore (0-1)",  text_cf["bertscore"],   GRAPHS_TEXT / "cf_vs_big_bertscore.png")
    bar_chart("ROUGE: Overlap-based Similarity",     "rouge",       "ROUGE-L (0-1)",    text_cf["rouge"],       GRAPHS_TEXT / "cf_vs_big_rouge.png")
    bar_chart("Readability: Flesch-Kincaid Grade",   "readability", "Score (FK/10)",    text_cf["readability"], GRAPHS_TEXT / "cf_vs_big_readability.png")
    grouped_bar("Cloudflare vs Big Players — Text Metrics",
                ["BERTScore", "ROUGE-L", "Readability"],
                ["bertscore", "rouge", "readability"],
                text_cf, GRAPHS_TEXT / "cf_vs_big_text_grouped.png")
    line_chart("Cloudflare vs Big Players — Text Performance",
               ["BERTScore", "ROUGE\n(0-1)", "Readability\n(0-10)"],
               ["bertscore", "rouge", "readability"],
               text_cf, GRAPHS_TEXT / "cf_vs_big_text_line.png")
else:
    print("\n[!] No text scores — run cloudflare_benchmark.py first to generate and score.")

# ── IMAGE graphs ──────────────────────────────────────────────────────────────
if image_cf:
    print(f"\nImage scores loaded: {image_cf}")
    print("Generating IMAGE graphs...")
    bar_chart("CLIP: Image-Text Alignment",  "clip",      "CLIP Score (0-1)", image_cf["clip"],      GRAPHS_IMAGE / "cf_vs_big_clip.png")
    bar_chart("Aesthetic: Image Quality",    "aesthetic", "Score (0-10)",     image_cf["aesthetic"], GRAPHS_IMAGE / "cf_vs_big_aesthetic.png")
    grouped_bar("Cloudflare vs Big Players — Image Metrics",
                ["CLIP Score", "Aesthetic"],
                ["clip", "aesthetic"],
                image_cf, GRAPHS_IMAGE / "cf_vs_big_image_grouped.png")
    line_chart("Cloudflare vs Big Players — Image Performance",
               ["CLIP Score", "Aesthetic\n(0-10)"],
               ["clip", "aesthetic"],
               image_cf, GRAPHS_IMAGE / "cf_vs_big_image_line.png")
else:
    print("\n[!] No image scores — run cloudflare_benchmark.py first to generate and score.")

# ── AUDIO graphs ──────────────────────────────────────────────────────────────
if audio_cf:
    print(f"\nAudio scores loaded: {audio_cf}")
    print("Generating AUDIO graphs...")
    bar_chart("CLAP: Audio-Text Alignment",  "clap", "CLAP Score (0-1)", audio_cf["clap"], GRAPHS_AUDIO / "cf_vs_big_clap.png")
    bar_chart("MOS: Mean Opinion Score",     "mos",  "Score (1-5)",      audio_cf["mos"],  GRAPHS_AUDIO / "cf_vs_big_mos.png")
    bar_chart("WER: Word Error Rate",        "wer",  "Error Rate (0-1)", audio_cf["wer"],  GRAPHS_AUDIO / "cf_vs_big_wer.png", lower_better=True)
    grouped_bar("Cloudflare vs Big Players — Audio Metrics",
                ["CLAP", "MOS", "WER"],
                ["clap", "mos", "wer"],
                audio_cf, GRAPHS_AUDIO / "cf_vs_big_audio_grouped.png",
                lower_better_keys=["wer"])
    line_chart("Cloudflare vs Big Players — Audio Performance",
               ["CLAP", "MOS\n(1-5)", "WER\n(lower=better)"],
               ["clap", "mos", "wer"],
               audio_cf, GRAPHS_AUDIO / "cf_vs_big_audio_line.png")
else:
    print("\n[!] No audio scores — run cloudflare_benchmark.py first to generate and score.")

print("\nDone. All graphs saved to cloudflare/outputs/graphs/")
