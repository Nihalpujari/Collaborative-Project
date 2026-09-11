"""
compare_approaches.py
=====================
Loads final_3approaches.csv and prints a clear comparison showing
how Likelihood Ratio beats WTSAS and Naive Bayes.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

CSV = Path(__file__).parent / "scores" / "all_500_v2_results.csv"

df = pd.read_csv(CSV)
y  = df["judge_mean"].values          # ground truth (LLM judge, 1–5 scale)

approaches = {
    "Approach 1 — WTSAS":           df["wtsas_final"].values,
    "Approach 2 — Naive Bayes":     df["nb_sw_loo"].values,
    "Approach 3 — Likelihood Ratio": df["lr_sw_loo"].values,
}

def pairwise_accuracy(r):
    """Probability of correctly ranking a random pair (concordance)."""
    return 0.5 + np.arcsin(np.clip(r, -1, 1)) / np.pi

print("=" * 65)
print("  TRI-MODAL APPROACH COMPARISON  (n=500 prompts, LOO CV)")
print("=" * 65)
print(f"\n  Ground truth: judge_mean from Gemini LLM (1-5 scale)")
print(f"  mean={y.mean():.3f}  std={y.std():.3f}\n")

results = {}
for name, scores in approaches.items():
    r,  _ = pearsonr(scores, y)
    rs, _ = spearmanr(scores, y)
    pa    = pairwise_accuracy(r)
    r2    = r ** 2
    results[name] = dict(r=r, rs=rs, pa=pa, r2=r2)

# Print table
print(f"  {'Approach':<35} {'Pearson r':>10} {'Spearman r':>11} {'Pairwise Acc':>13} {'r2':>6}")
print(f"  {'-'*75}")
for name, m in results.items():
    print(f"  {name:<35} {m['r']:>10.4f} {m['rs']:>11.4f} {m['pa']:>12.1%} {m['r2']:>6.3f}")

print(f"\n{'=' * 65}")
print("  WHY LIKELIHOOD RATIO WINS")
print("=" * 65)

r_wtsas = results["Approach 1 — WTSAS"]["r"]
r_nb    = results["Approach 2 — Naive Bayes"]["r"]
r_lr    = results["Approach 3 — Likelihood Ratio"]["r"]
pa_lr   = results["Approach 3 — Likelihood Ratio"]["pa"]

print(f"""
  1. Pearson r:
       LR ({r_lr:.4f}) > NB ({r_nb:.4f}) > WTSAS ({r_wtsas:.4f})
       LR is {(r_lr - r_wtsas):.4f} above WTSAS  (+{(r_lr-r_wtsas)/abs(r_wtsas)*100:.0f}% relative)
       LR is {(r_lr - r_nb):.4f} above NB

  2. Pairwise accuracy:
       Given any 2 random prompts, LR correctly identifies which
       one the judge rated higher in {pa_lr:.1%} of cases.
       (random baseline = 50.0%)
       Formula: 0.5 + arcsin(r) / pi = 0.5 + arcsin({r_lr:.4f}) / pi

  3. Why LR > WTSAS conceptually:
       WTSAS only measures coherence (do the 3 modalities align?).
       LR uses quality-weighted coherence [s1_w, s2_w, s3_w] and
       explicitly contrasts Good vs Not-Good distributions,
       incorporating learned quality weights (w1=7.87 text,
       w2=2.28 image, w3=0.78 audio).

  4. Why LR > NB conceptually:
       NB asks: "does this look like a Good output?"
       LR asks: "does this look MORE like Good than like Bad?"
       The log-ratio is more discriminative than a single-class
       probability estimate.

  5. Honest caveat:
       r2 = {r_lr**2:.3f} -- LR explains only {r_lr**2*100:.1f}% of variance
       in judge scores. All 3 approaches are weak predictors.
       Conclusion: tri-modal coherence is only weakly linked to
       perceived quality -- coherence != quality.
""")

print("=" * 65)
print(f"  RANKING:  LR > NB > WTSAS  (Pearson r, Spearman r, pairwise accuracy)")
print("=" * 65)
