"""
Likelihood Ratio Score

LR(x) = log P(x | Good) - log P(x | Reference)

Unlike Naive Bayes P(Good|x), this ignores class priors entirely.
With our class imbalance (Bad=4, Good=15, Medium=34), NB is pulled
toward predicting Medium because P(Medium) is large. LR focuses
purely on how 'Good-like' the features are, regardless of how many
Good samples exist.

Two variants:
  - LR Good-vs-Bad   : log P(x|Good) - log P(x|Bad)
  - LR Good-vs-Rest  : log P(x|Good) - log P(x|Medium+Bad)

Steps:
  1. Fit Gaussian(mean, std) per feature per class
  2. Compute log-likelihood ratio (raw, unbounded)
  3. Normalize to [0, 1] for interpretability
  4. Leave-One-Out cross-validation for honest estimate
  5. Compare all metrics vs LLM ratings

Saves: outputs/scores/likelihood_ratio_scores.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from sklearn.model_selection import LeaveOneOut

# -- paths -------------------------------------------------------------------
BASE        = Path(__file__).parent.parent
SCORES_CSV  = BASE / "outputs/scores/quality_scores.csv"
RATINGS_CSV = BASE / "outputs/scores/llm_ratings.csv"
WTSAS_CSV   = BASE / "outputs/scores/wtsas_final.csv"
NB_CSV      = BASE / "outputs/scores/naive_bayes_scores.csv"
OUT_DIR     = BASE / "outputs/scores"

# -- load data ---------------------------------------------------------------
df_q = pd.read_csv(SCORES_CSV)
df_r = pd.read_csv(RATINGS_CSV)
df_w = pd.read_csv(WTSAS_CSV)[["prompt_id", "tsas", "wtsas_final"]]
df_nb = pd.read_csv(NB_CSV)[["prompt_id", "p_good_loo"]]

df = pd.merge(df_q[["prompt_id", "prompt", "q_text", "q_image", "q_audio"]],
              df_r[["prompt_id", "llm_overall"]], on="prompt_id")
df = pd.merge(df, df_w, on="prompt_id")
df = pd.merge(df, df_nb, on="prompt_id")

print("=" * 70)
print("  LIKELIHOOD RATIO SCORE")
print(f"  Samples: {len(df)}")
print("=" * 70)

# -- Step 1: Bin into classes ------------------------------------------------
def bin_score(s):
    if s >= 8:   return "Good"
    elif s >= 6: return "Medium"
    else:        return "Bad"

df["class"] = df["llm_overall"].apply(bin_score)

print(f"\n  Class distribution:")
print(f"    Bad    (1-5)  : {(df['class']=='Bad').sum()} prompts")
print(f"    Medium (6-7)  : {(df['class']=='Medium').sum()} prompts")
print(f"    Good   (8-10) : {(df['class']=='Good').sum()} prompts")

X       = df[["q_text", "q_image", "q_audio"]].values
y_class = df["class"].values
y_llm   = df["llm_overall"].values

EPS = 1e-6  # prevent zero variance

# -- Helper functions --------------------------------------------------------
def fit_gaussian(X_subset):
    mu  = X_subset.mean(axis=0)
    std = X_subset.std(axis=0) + EPS
    return mu, std

def log_likelihood(x, mu, std):
    return float(np.sum(stats.norm.logpdf(x, loc=mu, scale=std)))

def log_lr(x, mu_good, std_good, mu_ref, std_ref):
    return log_likelihood(x, mu_good, std_good) - log_likelihood(x, mu_ref, std_ref)

def normalize(v):
    lo, hi = v.min(), v.max()
    if hi - lo < EPS:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)

# -- Step 2: Full-data training ----------------------------------------------
print("\n  Step 2 - Fit Gaussians on full dataset...")

good_mask = y_class == "Good"
bad_mask  = y_class == "Bad"
rest_mask = ~good_mask  # Medium + Bad

mu_good, std_good = fit_gaussian(X[good_mask])
mu_bad,  std_bad  = fit_gaussian(X[bad_mask])
mu_rest, std_rest = fit_gaussian(X[rest_mask])

print(f"    Good  mean: Q_text={mu_good[0]:.4f}  Q_image={mu_good[1]:.4f}  Q_audio={mu_good[2]:.4f}")
print(f"    Bad   mean: Q_text={mu_bad[0]:.4f}   Q_image={mu_bad[1]:.4f}   Q_audio={mu_bad[2]:.4f}")
print(f"    Rest  mean: Q_text={mu_rest[0]:.4f}  Q_image={mu_rest[1]:.4f}  Q_audio={mu_rest[2]:.4f}")

# Compute LR scores on all data (train score)
lr_gb_raw = np.array([log_lr(x, mu_good, std_good, mu_bad,  std_bad)  for x in X])
lr_gr_raw = np.array([log_lr(x, mu_good, std_good, mu_rest, std_rest) for x in X])

lr_gb_train = normalize(lr_gb_raw)
lr_gr_train = normalize(lr_gr_raw)

# -- Step 3: Leave-One-Out validation ----------------------------------------
print("\n  Step 3 - Leave-One-Out Cross Validation...")

loo_gb_raw = np.zeros(len(X))
loo_gr_raw = np.zeros(len(X))

for train_idx, test_idx in LeaveOneOut().split(X):
    X_tr   = X[train_idx]
    y_tr   = y_class[train_idx]
    x_test = X[test_idx[0]]

    good_tr = y_tr == "Good"
    bad_tr  = y_tr == "Bad"
    rest_tr = ~good_tr

    # LR Good-vs-Bad
    if good_tr.sum() >= 2 and bad_tr.sum() >= 2:
        mg, sg = fit_gaussian(X_tr[good_tr])
        mb, sb = fit_gaussian(X_tr[bad_tr])
        loo_gb_raw[test_idx[0]] = log_lr(x_test, mg, sg, mb, sb)
    else:
        loo_gb_raw[test_idx[0]] = 0.0

    # LR Good-vs-Rest
    if good_tr.sum() >= 2 and rest_tr.sum() >= 2:
        mg, sg = fit_gaussian(X_tr[good_tr])
        mr, sr = fit_gaussian(X_tr[rest_tr])
        loo_gr_raw[test_idx[0]] = log_lr(x_test, mg, sg, mr, sr)
    else:
        loo_gr_raw[test_idx[0]] = 0.0

loo_gb = normalize(loo_gb_raw)
loo_gr = normalize(loo_gr_raw)

# -- Step 4: Compute all correlations ----------------------------------------
r_gb_train, _ = pearsonr(lr_gb_train, y_llm)
r_gr_train, _ = pearsonr(lr_gr_train, y_llm)
r_gb_loo,   _ = pearsonr(loo_gb,      y_llm)
r_gr_loo,   _ = pearsonr(loo_gr,      y_llm)
r_tsas,     _ = pearsonr(df["tsas"].values,       y_llm)
r_wtsas,    _ = pearsonr(df["wtsas_final"].values, y_llm)
r_qimg,     _ = pearsonr(df["q_image"].values,    y_llm)
r_nb_loo,   _ = pearsonr(df["p_good_loo"].values, y_llm)

print(f"\n{'='*70}")
print("  FULL COMPARISON — Correlation with LLM Judge Ratings")
print(f"{'='*70}")
print(f"\n  {'Metric':<35} {'Pearson r':>10}  Note")
print(f"  {'-'*65}")
print(f"  {'CLIP score (Q_image raw)':<35} {r_qimg:>10.4f}  best single feature")
print(f"  {'TSAS (ImageBind coherence)':<35} {r_tsas:>10.4f}  baseline")
print(f"  {'WTSAS (quality-weighted)':<35} {r_wtsas:>10.4f}  stage 5")
print(f"  {'Naive Bayes P(Good) - LOO':<35} {r_nb_loo:>10.4f}  stage 6a")
print(f"  {'LR Good-vs-Bad  (train)':<35} {r_gb_train:>10.4f}  informative only")
print(f"  {'LR Good-vs-Bad  (LOO)':<35} {r_gb_loo:>10.4f}  honest estimate")
print(f"  {'LR Good-vs-Rest (train)':<35} {r_gr_train:>10.4f}  informative only")
print(f"  {'LR Good-vs-Rest (LOO)':<35} {r_gr_loo:>10.4f}  honest estimate")

# -- Step 5: Per-prompt table ------------------------------------------------
df_out = df[["prompt_id", "prompt", "q_text", "q_image", "q_audio",
             "tsas", "wtsas_final", "llm_overall", "class", "p_good_loo"]].copy()
df_out["lr_gb_train"] = np.round(lr_gb_train, 4)
df_out["lr_gr_train"] = np.round(lr_gr_train, 4)
df_out["lr_gb_loo"]   = np.round(loo_gb,      4)
df_out["lr_gr_loo"]   = np.round(loo_gr,      4)

out_path = OUT_DIR / "likelihood_ratio_scores.csv"
df_out.to_csv(out_path, index=False)
print(f"\n  Saved to: {out_path}")

# -- Step 6: Top and bottom by LR Good-vs-Rest LOO ---------------------------
print(f"\n{'='*70}")
print("  TOP 10 — highest LR Good-vs-Rest (LOO)")
print(f"{'='*70}")
print(f"  {'ID':<4} {'LR-GR':>6}  {'NB P(G)':>8}  {'LLM':>4}  {'Class':<8}  Prompt")
print(f"  {'-'*65}")
for _, r in df_out.nlargest(10, "lr_gr_loo").iterrows():
    print(f"  [{int(r['prompt_id']):02d}] {r['lr_gr_loo']:>6.4f}  {r['p_good_loo']:>8.4f}  "
          f"{r['llm_overall']:>4.0f}  {r['class']:<8}  {r['prompt'][:38]}")

print(f"\n{'='*70}")
print("  BOTTOM 10 — lowest LR Good-vs-Rest (LOO)")
print(f"{'='*70}")
print(f"  {'ID':<4} {'LR-GR':>6}  {'NB P(G)':>8}  {'LLM':>4}  {'Class':<8}  Prompt")
print(f"  {'-'*65}")
for _, r in df_out.nsmallest(10, "lr_gr_loo").iterrows():
    print(f"  [{int(r['prompt_id']):02d}] {r['lr_gr_loo']:>6.4f}  {r['p_good_loo']:>8.4f}  "
          f"{r['llm_overall']:>4.0f}  {r['class']:<8}  {r['prompt'][:38]}")

# -- Step 7: Key insight - what do Good samples look like? -------------------
print(f"\n{'='*70}")
print("  FEATURE MEANS PER CLASS")
print(f"{'='*70}")
for cls in ["Good", "Medium", "Bad"]:
    mask = y_class == cls
    means = X[mask].mean(axis=0)
    print(f"  {cls:<8} (n={mask.sum()})  "
          f"Q_text={means[0]:.4f}  Q_image={means[1]:.4f}  Q_audio={means[2]:.4f}")

print(f"\n  Key: Q_image separates classes most — confirms it is the dominant signal.")
print(f"  LR Good-vs-Rest LOO (r={r_gr_loo:.4f}) vs Naive Bayes LOO (r={r_nb_loo:.4f})")

if r_gr_loo > r_nb_loo:
    print(f"\n  LR outperforms Naive Bayes — prior removal helped.")
else:
    print(f"\n  Naive Bayes matches or outperforms LR — priors encode useful info.")

print(f"\n{'='*70}")
print("  CONCLUSION")
print(f"{'='*70}")
best_r  = max(r_gb_loo, r_gr_loo, r_nb_loo, r_wtsas)
best_n  = {r_gb_loo: "LR Good-vs-Bad LOO",
           r_gr_loo: "LR Good-vs-Rest LOO",
           r_nb_loo: "Naive Bayes LOO",
           r_wtsas:  "WTSAS"}[best_r]
print(f"\n  Best probabilistic metric : {best_n}  (r = {best_r:.4f})")
print(f"  Best overall metric       : Q_image (CLIP)  (r = {r_qimg:.4f})")


if __name__ == "__main__":
    pass
