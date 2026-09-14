"""
Naive Bayes Score  —  v2 (works with full_pipeline.py output)
=============================================================

Input : complete_scores.csv  (output of full_pipeline.py)
        OR any CSV with columns: q_text, q_image, q_audio, llm_overall

Usage:
    python naive_bayes_score_v2.py --input complete_scores.csv
    python naive_bayes_score_v2.py          (defaults to complete_scores.csv)
"""

import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut

parser = argparse.ArgumentParser()
parser.add_argument("--input",  default="complete_scores.csv")
parser.add_argument("--output", default="naive_bayes_results.csv")
args = parser.parse_args()

df = pd.read_csv(args.input)

print("=" * 60)
print("  NAIVE BAYES QUALITY SCORE")
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

# -- Step 2: train on all data -----------------------------------------------
gnb   = GaussianNB()
gnb.fit(X, y_class)
classes  = list(gnb.classes_)
good_idx = classes.index("Good")

p_good_train = gnb.predict_proba(X)[:, good_idx]

# -- Step 3: LOO validation --------------------------------------------------
print("\n  Running Leave-One-Out cross-validation...")
p_good_loo = np.zeros(len(X))

for train_idx, test_idx in LeaveOneOut().split(X):
    gnb_loo = GaussianNB()
    gnb_loo.fit(X[train_idx], y_class[train_idx])
    classes_loo = list(gnb_loo.classes_)
    if "Good" not in classes_loo:
        p_good_loo[test_idx[0]] = 0.0
    else:
        gi = classes_loo.index("Good")
        p_good_loo[test_idx[0]] = gnb_loo.predict_proba(X[test_idx])[ 0, gi]

# -- Step 4: correlations ----------------------------------------------------
r_train, _ = pearsonr(p_good_train, y_llm)
r_loo,   _ = pearsonr(p_good_loo,   y_llm)
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
print(f"  {'Naive Bayes (train)':<30} {r_train:>8.4f}  <- overfit, informative only")
print(f"  {'Naive Bayes (LOO)':<30} {r_loo:>8.4f}  <- honest estimate")

# -- Step 5: save ------------------------------------------------------------
df_out = df.copy()
df_out["p_good_train"] = np.round(p_good_train, 4)
df_out["p_good_loo"]   = np.round(p_good_loo,   4)

df_out.to_csv(args.output, index=False)
print(f"\n  Saved to: {args.output}")

# -- Step 6: top / bottom --------------------------------------------------
print(f"\n  TOP 10 — highest P(Good) LOO")
print(f"  {'ID':<5} {'P(Good)':>8}  {'LLM':>4}  {'Class':<8}  Prompt")
print(f"  {'-'*55}")
for _, r in df_out.nlargest(10, "p_good_loo").iterrows():
    pid  = int(r["prompt_id"]) if "prompt_id" in r else "-"
    pmpt = str(r["prompt"])[:40] if "prompt" in r else ""
    print(f"  [{pid:<3}] {r['p_good_loo']:>8.4f}  {r['llm_overall']:>4.0f}  {r['class']:<8}  {pmpt}")

print(f"\n  BOTTOM 10 — lowest P(Good) LOO")
print(f"  {'ID':<5} {'P(Good)':>8}  {'LLM':>4}  {'Class':<8}  Prompt")
print(f"  {'-'*55}")
for _, r in df_out.nsmallest(10, "p_good_loo").iterrows():
    pid  = int(r["prompt_id"]) if "prompt_id" in r else "-"
    pmpt = str(r["prompt"])[:40] if "prompt" in r else ""
    print(f"  [{pid:<3}] {r['p_good_loo']:>8.4f}  {r['llm_overall']:>4.0f}  {r['class']:<8}  {pmpt}")

print(f"\n  Done. Use 'p_good_loo' column as the Naive Bayes final score.")
