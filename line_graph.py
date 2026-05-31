"""
LINE GRAPH - Model Performance Comparison
Small Models vs Big Players across all metrics
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import csv

# Load config
_cfg = {}
exec(open("config.py").read(), _cfg)
BIG_PLAYERS = _cfg["BIG_PLAYERS_BENCHMARKS"]

# Colors for each model
COLORS = {
    "mistral":  "#6C5CE7",
    "groq":     "#00B894",
    "cerebras": "#FDCB6E",
    "claude":   "#0984E3",
    "chatgpt":  "#00CEC9",
    "gemini":   "#D63031",
}

MARKERS = {
    "mistral":  "o",
    "groq":     "s",
    "cerebras": "^",
    "claude":   "D",
    "chatgpt":  "P",
    "gemini":   "*",
}

def load_small_model_scores():
    """Load scores from CSV if exists, else use example values"""
    csv_path = Path("outputs/text_benchmark/scores/text_scores.csv")
    scores = {}
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                scores[row["model"]] = {
                    "bertscore":   float(row["bertscore"]),
                    "rouge":       float(row["rouge"]),
                    "readability": float(row["readability"]),
                }
        print(f"✅ Loaded scores from {csv_path}")
    else:
        # Example values if benchmark not yet run
        scores = {
            "mistral":  {"bertscore": 0.82, "rouge": 0.36, "readability": 7.4},
            "groq":     {"bertscore": 0.81, "rouge": 0.34, "readability": 7.1},
            "cerebras": {"bertscore": 0.80, "rouge": 0.33, "readability": 6.9},
        }
        print("⚠️  Using example scores. Run text_benchmark_full.py first for real scores.")
    return scores


def make_line_graph(small_scores):
    """
    Line graph: X = metrics, Y = score value
    One line per model (small + big players)
    """
    metrics      = ["BERTScore", "ROUGE", "Readability\n(0-10)"]
    metric_keys  = ["bertscore", "rouge", "readability"]

    all_models = {**small_scores}
    # Add big players
    for bp, bv in BIG_PLAYERS.items():
        all_models[bp] = {k: bv[k] for k in metric_keys if k in bv}

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(metrics))

    for model, scores in all_models.items():
        vals = [scores.get(k, 0) for k in metric_keys]
        is_big = model in BIG_PLAYERS
        ax.plot(
            x, vals,
            marker=MARKERS.get(model, "o"),
            color=COLORS.get(model, "#888"),
            linewidth=2.5 if is_big else 2.0,
            linestyle="--" if is_big else "-",
            markersize=9 if is_big else 8,
            label=f"{model.capitalize()} {'(Big Player)' if is_big else ''}",
            zorder=3
        )
        # Label each point
        for xi, v in zip(x, vals):
            ax.annotate(
                f"{v:.2f}",
                (xi, v),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7.5,
                color=COLORS.get(model, "#888")
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Text Model Performance — Small Models vs Big Players",
        fontsize=14, fontweight="bold", pad=14
    )
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", alpha=0.15)

    # Shade region: small models area
    sm_vals = np.array([[s.get(k, 0) for k in metric_keys]
                         for s in small_scores.values()])
    if len(sm_vals) > 0:
        ax.fill_between(
            x,
            sm_vals.min(axis=0),
            sm_vals.max(axis=0),
            alpha=0.08, color="#6C5CE7", label="_nolegend_"
        )

    plt.tight_layout()

    out = Path("outputs/text_benchmark/graphs")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "line_graph_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Line graph saved → {path}")
    return path


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GENERATING LINE GRAPH — Model Performance Comparison")
    print("="*70 + "\n")

    small = load_small_model_scores()
    make_line_graph(small)

    print("\n✅ Done! Open outputs/text_benchmark/graphs/line_graph_comparison.png")
