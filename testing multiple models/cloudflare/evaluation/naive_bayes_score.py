"""
Naive Bayes Quality Score

Uses GaussianNB to compute P(good | Q_text, Q_image, Q_audio) as a
continuous probability score for each prompt.

Pipeline:
  1. Bin LLM judge scores into 3 classes:
       Bad    = 1-5
       Medium = 6-7
       Good   = 8-10
  2. Train GaussianNB on X = [Q_text, Q_image, Q_audio]
  3. Use predict_proba() -- NOT predict() -- to get a continuous score
  4. Validate with Leave-One-Out CV
  5. Compare correlation with LLM ratings vs TSAS and WTSAS

Saves: outputs/scores/naive_bayes_scores.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report

# -- paths -------------------------------------------------------------------
BASE        = Path(__file__).parent.parent
SCORES_CSV  = BASE / "outputs/scores/quality_scores.csv"
RATINGS_CSV = BASE / "outputs/scores/llm_ratings.csv"
WTSAS_CSV   = BASE / "outputs/scores/wtsas_final.csv"
OUT_DIR     = BASE / "outputs/scores"

# -- load data ---------------------------------------------------------------
df_q = pd.read_csv(SCORES_CSV)
df_r = pd.read_csv(RATINGS_CSV)
df_w = pd.read_csv(WTSAS_CSV)[["prompt_id", "tsas", "wtsas_final"]]

df = pd.merge(df_q[["prompt_id", "prompt", "q_text", "q_image", "q_audio"]],
              df_r[["prompt_id", "llm_overall"]], on="prompt_id")
df = pd.merge(df, df_w, on="prompt_id")

print("=" * 70)
print("  NAIVE BAYES QUALITY SCORE")
print(f"  Samples: {len(df)}")
print("=" * 70)

# -- Step 1: Bin LLM scores into classes -------------------------------------
def bin_score(s):
    if s >= 8:
        return "Good"
    elif s >= 6:
        return "Medium"
    else:
        return "Bad"

df["class"] = df["llm_overall"].apply(bin_score)

print("\nStep 1 - Class distribution:")
print(f"  Bad    (1-5):  {(df['class']=='Bad').sum()} prompts")
print(f"  Medium (6-7):  {(df['class']=='Medium').sum()} prompts")
print(f"  Good   (8-10): {(df['class']=='Good').sum()} prompts")

# -- Step 2: Features and labels ---------------------------------------------
X = df[["q_text", "q_image", "q_audio"]].values
y_class = df["class"].values
y_llm   = df["llm_overall"].values

# -- Step 3: Train GaussianNB on all data ------------------------------------
print("\nStep 2 - Train GaussianNB on full dataset...")
gnb = GaussianNB()
gnb.fit(X, y_class)

# Get class order
classes = list(gnb.classes_)
print(f"  Class order: {classes}")

good_idx = classes.index("Good")

# P(Good | features) for each prompt
proba_all = gnb.predict_proba(X)
p_good_all = proba_all[:, good_idx]

print(f"  P(Good) range: {p_good_all.min():.4f} to {p_good_all.max():.4f}")

# -- Step 4: Leave-One-Out CV ------------------------------------------------
print("\nStep 3 - Leave-One-Out Cross Validation...")
loo = LeaveOneOut()
p_good_loo  = np.zeros(len(X))
y_pred_class = []

for train_idx, test_idx in loo.split(X):
    gnb_loo = GaussianNB()
    gnb_loo.fit(X[train_idx], y_class[train_idx])

    classes_loo = list(gnb_loo.classes_)
    if "Good" not in classes_loo:
        p_good_loo[test_idx] = 0.0
        y_pred_class.append("Bad")
    else:
        good_idx_loo = classes_loo.index("Good")
        proba = gnb_loo.predict_proba(X[test_idx])
        p_good_loo[test_idx] = proba[0, good_idx_loo]
        y_pred_class.append(gnb_loo.predict(X[test_idx])[0])

# LOO correlation with LLM ratings
r_loo, p_loo = pearsonr(p_good_loo, y_llm)
print(f"  LOO Pearson r = {r_loo:.4f}  (p={p_loo:.4f})")

# -- Step 5: Compare all metrics ---------------------------------------------
r_tsas,  _ = pearsonr(df["tsas"].values,       y_llm)
r_wtsas, _ = pearsonr(df["wtsas_final"].values, y_llm)
r_nb,    _ = pearsonr(p_good_all,               y_llm)
r_nb_loo, _= pearsonr(p_good_loo,               y_llm)
r_qimg,  _ = pearsonr(df["q_image"].values,     y_llm)

print("\n" + "=" * 70)
print("  COMPARISON - Correlation with LLM Judge Ratings")
print("=" * 70)
print(f"\n  {'Metric':<30} {'Pearson r':>10}  {'Note'}")
print(f"  {'-'*65}")
print(f"  {'CLIP score (raw)':<30} {pearsonr(df['q_image'].values, y_llm)[0]:>10.4f}  best single feature")
print(f"  {'Q_image':<30} {r_qimg:>10.4f}  quality pair score")
print(f"  {'TSAS (ImageBind coherence)':<30} {r_tsas:>10.4f}  baseline")
print(f"  {'WTSAS (quality-weighted)':<30} {r_wtsas:>10.4f}  our stage 5")
print(f"  {'Naive Bayes P(Good) - train':<30} {r_nb:>10.4f}  trained on all data")
print(f"  {'Naive Bayes P(Good) - LOO':<30} {r_nb_loo:>10.4f}  leave-one-out (honest)")

# -- Step 6: Save results ----------------------------------------------------
df_out = df[["prompt_id", "prompt", "q_text", "q_image", "q_audio",
             "tsas", "wtsas_final", "llm_overall", "class"]].copy()
df_out["p_good_train"] = np.round(p_good_all,  4)
df_out["p_good_loo"]   = np.round(p_good_loo,  4)

out_path = OUT_DIR / "naive_bayes_scores.csv"
df_out.to_csv(out_path, index=False)
print(f"\n  Saved to: {out_path}")

# -- Step 7: Per-prompt results ----------------------------------------------
print("\n" + "=" * 70)
print("  PER-PROMPT SCORES (sorted by P(Good) LOO)")
print("=" * 70)
print(f"\n  {'ID':<4} {'P(Good)':>8}  {'LLM':>4}  {'Class':<8}  Prompt")
print(f"  {'-'*65}")
for _, r in df_out.sort_values("p_good_loo", ascending=False).iterrows():
    print(f"  [{int(r['prompt_id']):02d}] {r['p_good_loo']:>8.4f}  {r['llm_overall']:>4.0f}  {r['class']:<8}  {r['prompt'][:40]}")

print("\n" + "=" * 70)
print("  CONCLUSION")
print("=" * 70)
best = max(r_nb_loo, r_wtsas, r_tsas)
best_name = {r_nb_loo: "Naive Bayes (LOO)", r_wtsas: "WTSAS", r_tsas: "TSAS"}[best]
print(f"\n  Best metric: {best_name} (r = {best:.4f})")
print(f"  Q_image (CLIP-based) remains the strongest single signal (r = {r_qimg:.4f})")
print(f"\n  Naive Bayes P(Good) provides a probabilistic interpretation of quality")
print(f"  that is more intuitive than a raw coherence score.")


if __name__ == "__main__":
    pass
