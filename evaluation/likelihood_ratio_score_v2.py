"""
Likelihood Ratio Score  —  v2 (works with full_pipeline.py output)
==================================================================

LR(x) = log P(x | Good) - log P(x | Rest)

Input : complete_scores.csv  (output of full_pipeline.py)
        OR any CSV with columns: q_text, q_image, q_audio, llm_overall

Usage:
    python likelihood_ratio_score_v2.py --input complete_scores.csv
    python likelihood_ratio_score_v2.py          (defaults to complete_scores.csv)
"""

import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from sklearn.model_selection import LeaveOneOut

parser = argparse.ArgumentParser()
parser.add_argument("--input",  default="complete_scores.csv")
parser.add_argument("--output", default="likelihood_ratio_results.csv")
args = parser.parse_args()

df = pd.read_csv(args.input)

print("=" * 60)
print("  LIKELIHOOD RATIO SCORE")
print(f"  Input  : {args.input}")
print(f"  Samples: {len(df)}")
print("=" * 60)

# -- Step 1: bin LLM scores --------------------------------------------------
def bin_score(s):
    if s >= 8:   return "Good"
    elif s >= 6: return "Medium"
    else:        return "Bad"

df["class"] = df["llm_overall"].apply(bin_score)

print(f"\n  Class distribution:")
print(f"    Bad    (1-5)  : {(df['class']=='Bad').sum()}")
print(f"    Medium (6-7)  : {(df['class']=='Medium').sum()}")
print(f"    Good   (8-10) : {(df['class']=='Good').sum()}")

X       = df[["q_text", "q_image", "q_audio"]].values
y_class = df["class"].values
y_llm   = df["llm_overall"].values

EPS = 1e-6

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

# -- Step 2: full-data training ----------------------------------------------
print("\n  Fitting Gaussians on full dataset...")
good_mask = y_class == "Good"
rest_mask = ~good_mask

mu_good, std_good = fit_gaussian(X[good_mask])
mu_rest, std_rest = fit_gaussian(X[rest_mask])

print(f"  Good mean: Q_text={mu_good[0]:.4f}  Q_image={mu_good[1]:.4f}  Q_audio={mu_good[2]:.4f}")
print(f"  Rest mean: Q_text={mu_rest[0]:.4f}  Q_image={mu_rest[1]:.4f}  Q_audio={mu_rest[2]:.4f}")

lr_train_raw = np.array([log_lr(x, mu_good, std_good, mu_rest, std_rest) for x in X])
lr_train     = normalize(lr_train_raw)

# -- Step 3: LOO validation --------------------------------------------------
print("\n  Running Leave-One-Out cross-validation...")
lr_loo_raw = np.zeros(len(X))

for train_idx, test_idx in LeaveOneOut().split(X):
    X_tr  = X[train_idx]
    y_tr  = y_class[train_idx]
    x_t   = X[test_idx[0]]

    good_tr = y_tr == "Good"
    rest_tr = ~good_tr

    if good_tr.sum() >= 2 and rest_tr.sum() >= 2:
        mg, sg = fit_gaussian(X_tr[good_tr])
        mr, sr = fit_gaussian(X_tr[rest_tr])
        lr_loo_raw[test_idx[0]] = log_lr(x_t, mg, sg, mr, sr)
    else:
        lr_loo_raw[test_idx[0]] = 0.0

lr_loo = normalize(lr_loo_raw)

# -- Step 4: correlations ----------------------------------------------------
r_train, _ = pearsonr(lr_train, y_llm)
r_loo,   _ = pearsonr(lr_loo,   y_llm)
r_qimg,  _ = pearsonr(df["q_image"].values, y_llm)

has_tsas  = "tsas"        in df.columns
has_wtsas = "wtsas_final" in df.columns
r_tsas    = pearsonr(df["tsas"].values,        y_llm)[0] if has_tsas  else float("nan")
r_wtsas   = pearsonr(df["wtsas_final"].values, y_llm)[0] if has_wtsas else float("nan")

print(f"\n{'='*60}")
print("  CORRELATION WITH LLM RATINGS")
print(f"{'='*60}")
print(f"\n  {'Metric':<30} {'r':>8}")
print(f"  {'-'*40}")
print(f"  {'Q_image (CLIP)':<30} {r_qimg:>8.4f}")
if has_tsas:
    print(f"  {'TSAS (ImageBind)':<30} {r_tsas:>8.4f}")
if has_wtsas:
    print(f"  {'WTSAS (final)':<30} {r_wtsas:>8.4f}")
print(f"  {'LR Good-vs-Rest (train)':<30} {r_train:>8.4f}  <- overfit, informative only")
print(f"  {'LR Good-vs-Rest (LOO)':<30} {r_loo:>8.4f}  <- honest estimate")

# -- Step 5: save ------------------------------------------------------------
df_out = df.copy()
df_out["lr_gr_train"] = np.round(lr_train, 4)
df_out["lr_gr_loo"]   = np.round(lr_loo,   4)

df_out.to_csv(args.output, index=False)
print(f"\n  Saved to: {args.output}")

# -- Step 6: top / bottom ----------------------------------------------------
print(f"\n  TOP 10 — highest LR score (LOO)")
print(f"  {'ID':<5} {'LR-GR':>7}  {'LLM':>4}  {'Class':<8}  Prompt")
print(f"  {'-'*55}")
for _, r in df_out.nlargest(10, "lr_gr_loo").iterrows():
    pid  = int(r["prompt_id"]) if "prompt_id" in r else "-"
    pmpt = str(r["prompt"])[:40] if "prompt" in r else ""
    print(f"  [{pid:<3}] {r['lr_gr_loo']:>7.4f}  {r['llm_overall']:>4.0f}  {r['class']:<8}  {pmpt}")

print(f"\n  BOTTOM 10 — lowest LR score (LOO)")
print(f"  {'ID':<5} {'LR-GR':>7}  {'LLM':>4}  {'Class':<8}  Prompt")
print(f"  {'-'*55}")
for _, r in df_out.nsmallest(10, "lr_gr_loo").iterrows():
    pid  = int(r["prompt_id"]) if "prompt_id" in r else "-"
    pmpt = str(r["prompt"])[:40] if "prompt" in r else ""
    print(f"  [{pid:<3}] {r['lr_gr_loo']:>7.4f}  {r['llm_overall']:>4.0f}  {r['class']:<8}  {pmpt}")

print(f"\n  Done. Use 'lr_gr_loo' column as the Likelihood Ratio final score.")
