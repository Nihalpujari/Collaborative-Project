"""
Stage 5 - Quality-Weighted Coherence Score (WTSAS)

Uses learned weights (w1, w2, w3) from Linear Regression and
pairwise coherence scores (s1, s2, s3) from ImageBind to compute
the final Quality-Weighted Tri-modal Alignment Score (WTSAS).

Formula:
  s1_weighted = s1 x (w1*Q_text  + w2*Q_image) / (w1 + w2)
  s2_weighted = s2 x (w1*Q_text  + w3*Q_audio) / (w1 + w3)
  s3_weighted = s3 x (w2*Q_image + w3*Q_audio) / (w2 + w3)

  WTSAS = (s1_weighted + s2_weighted + s3_weighted) / 3
  Final = WTSAS - lambda x variance(s1_weighted, s2_weighted, s3_weighted)

Also computes plain TSAS for comparison to show WTSAS improvement.

Saves: outputs/scores/wtsas_final.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# -- paths -------------------------------------------------------------------
BASE         = Path(__file__).parent.parent
SCORES_CSV   = BASE / "outputs/scores/quality_scores.csv"
RATINGS_CSV  = BASE / "outputs/scores/llm_ratings.csv"
IB_CSV       = BASE / "outputs/scores/imagebind_coherence_scores.csv"
WEIGHTS_CSV  = BASE / "outputs/scores/weights_summary.csv"
OUT_DIR      = BASE / "outputs/scores"

# -- load data ---------------------------------------------------------------
df_q   = pd.read_csv(SCORES_CSV)
df_r   = pd.read_csv(RATINGS_CSV)
df_ib  = pd.read_csv(IB_CSV)
df_w   = pd.read_csv(WEIGHTS_CSV)

# Merge all on prompt_id
df = pd.merge(df_q[["prompt_id", "q_text", "q_image", "q_audio"]], df_ib, on="prompt_id")
df = pd.merge(df, df_r[["prompt_id", "llm_overall"]], on="prompt_id")

# -- learned weights ---------------------------------------------------------
w1 = float(df_w["w1_q_text"].iloc[0])
w2 = float(df_w["w2_q_image"].iloc[0])
w3 = float(df_w["w3_q_audio"].iloc[0])
LAMBDA = float(df_w["optimal_lambda"].iloc[0])

print("=" * 70)
print("  STAGE 5 - Quality-Weighted Coherence Score (WTSAS)")
print("=" * 70)
print(f"\n  Learned weights:  w1(text)={w1}  w2(image)={w2}  w3(audio)={w3}")
print(f"  Lambda:           {LAMBDA}")
print(f"  Samples:          {len(df)}\n")

# -- compute WTSAS -----------------------------------------------------------
results = []

for _, row in df.iterrows():
    s1 = row["s1_text_image"]
    s2 = row["s2_text_audio"]
    s3 = row["s3_image_audio"]

    Q_text  = row["q_text"]
    Q_image = row["q_image"]
    Q_audio = row["q_audio"]

    # Quality-weighted coherence scores
    s1_w = s1 * (w1*Q_text  + w2*Q_image) / (w1 + w2)
    s2_w = s2 * (w1*Q_text  + w3*Q_audio) / (w1 + w3)
    s3_w = s3 * (w2*Q_image + w3*Q_audio) / (w2 + w3)

    wtsas    = (s1_w + s2_w + s3_w) / 3
    variance = float(np.var([s1_w, s2_w, s3_w]))
    final    = wtsas - LAMBDA * variance

    # Plain TSAS for comparison
    tsas = row["tsas"]

    results.append({
        "prompt_id":   int(row["prompt_id"]),
        "prompt":      row["prompt"][:80],
        "s1":          round(s1, 4),
        "s2":          round(s2, 4),
        "s3":          round(s3, 4),
        "s1_weighted": round(s1_w, 4),
        "s2_weighted": round(s2_w, 4),
        "s3_weighted": round(s3_w, 4),
        "tsas":        round(tsas, 4),
        "wtsas":       round(wtsas, 4),
        "wtsas_final": round(final, 4),
        "llm_overall": row["llm_overall"],
    })

df_out = pd.DataFrame(results)

# -- save --------------------------------------------------------------------
out_path = OUT_DIR / "wtsas_final.csv"
df_out.to_csv(out_path, index=False)
print(f"  Saved to: {out_path}\n")

# -- compare TSAS vs WTSAS correlation with LLM ratings ---------------------
y = df_out["llm_overall"].values

r_tsas,  _ = pearsonr(df_out["tsas"].values,       y)
r_wtsas, _ = pearsonr(df_out["wtsas"].values,       y)
r_final, _ = pearsonr(df_out["wtsas_final"].values, y)

print("=" * 70)
print("  COMPARISON — Correlation with LLM Judge Ratings")
print("=" * 70)
print(f"\n  {'Metric':<20} {'Pearson r':>10}  {'Interpretation'}")
print(f"  {'-'*60}")
print(f"  {'TSAS (baseline)':<20} {r_tsas:>10.4f}  plain avg of s1,s2,s3")
print(f"  {'WTSAS':<20} {r_wtsas:>10.4f}  quality-weighted coherence")
print(f"  {'WTSAS + penalty':<20} {r_final:>10.4f}  WTSAS - lambda x variance")

improvement = r_final - r_tsas
print(f"\n  Improvement over TSAS: {improvement:+.4f}")
if improvement > 0:
    print(f"  WTSAS outperforms plain TSAS -- quality weighting helps.")
else:
    print(f"  TSAS outperforms WTSAS -- investigate weight calibration.")

# -- summary stats -----------------------------------------------------------
print("\n" + "=" * 70)
print("  SCORE DISTRIBUTIONS")
print("=" * 70)
for col, label in [("tsas", "TSAS"), ("wtsas", "WTSAS"), ("wtsas_final", "WTSAS+penalty")]:
    print(f"  {label:<20} avg={df_out[col].mean():.4f}  "
          f"min={df_out[col].min():.4f}  max={df_out[col].max():.4f}  "
          f"std={df_out[col].std():.4f}")

# -- top and bottom prompts --------------------------------------------------
print("\n" + "=" * 70)
print("  TOP 5 — highest WTSAS final score")
print("=" * 70)
for _, r in df_out.nlargest(5, "wtsas_final").iterrows():
    print(f"  [{r['prompt_id']:02d}] {r['prompt'][:55]}")
    print(f"       TSAS={r['tsas']}  WTSAS={r['wtsas_final']}  LLM={r['llm_overall']}/10\n")

print("  BOTTOM 5 — lowest WTSAS final score")
print("=" * 70)
for _, r in df_out.nsmallest(5, "wtsas_final").iterrows():
    print(f"  [{r['prompt_id']:02d}] {r['prompt'][:55]}")
    print(f"       TSAS={r['tsas']}  WTSAS={r['wtsas_final']}  LLM={r['llm_overall']}/10\n")


if __name__ == "__main__":
    pass
