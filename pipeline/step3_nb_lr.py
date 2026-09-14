"""
STEP 3 — Naive Bayes + Likelihood Ratio Scoring
================================================

Reads : scores/complete_scores.csv  (from step2_full_pipeline.py)
Runs  : Naive Bayes P(Good) + Likelihood Ratio (Good-vs-Rest)
Saves : scores/naive_bayes_results.csv
        scores/likelihood_ratio_results.csv
        scores/final_comparison.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut

HERE     = Path(__file__).parent
IN_CSV   = HERE / "scores/complete_scores.csv"
SCORES   = HERE / "scores"

df = pd.read_csv(IN_CSV)

print("=" * 60)
print("  STEP 3 — Naive Bayes + Likelihood Ratio")
print(f"  Samples: {len(df)}")
print("=" * 60)

# -- bin LLM scores ----------------------------------------------------------
def bin_score(s):
    if s >= 8:   return "Good"
    elif s >= 6: return "Medium"
    else:        return "Bad"

df["class"] = df["llm_overall"].apply(bin_score)
y_llm = df["llm_overall"].values
X     = df[["q_text","q_image","q_audio"]].values
y_cls = df["class"].values

print(f"\n  Class distribution:")
print(f"    Bad    : {(df['class']=='Bad').sum()}")
print(f"    Medium : {(df['class']=='Medium').sum()}")
print(f"    Good   : {(df['class']=='Good').sum()}")


# ============================================================
#  NAIVE BAYES
# ============================================================
print("\n" + "─" * 60)
print("  NAIVE BAYES")
print("─" * 60)

gnb   = GaussianNB().fit(X, y_cls)
cidx  = list(gnb.classes_).index("Good")
p_train = gnb.predict_proba(X)[:, cidx]

print("  Running LOO...")
p_loo = np.zeros(len(X))
for tr, te in LeaveOneOut().split(X):
    g = GaussianNB().fit(X[tr], y_cls[tr])
    cl = list(g.classes_)
    p_loo[te[0]] = g.predict_proba(X[te])[ 0, cl.index("Good")] if "Good" in cl else 0.0

r_nb_tr, _ = pearsonr(p_train, y_llm)
r_nb_lo, _ = pearsonr(p_loo,   y_llm)
print(f"  Pearson r (train) = {r_nb_tr:.4f}")
print(f"  Pearson r (LOO)   = {r_nb_lo:.4f}")


# ============================================================
#  LIKELIHOOD RATIO  (Good vs Rest)
# ============================================================
print("\n" + "─" * 60)
print("  LIKELIHOOD RATIO (Good-vs-Rest)")
print("─" * 60)

EPS = 1e-6

def fit_gauss(Xs):
    return Xs.mean(axis=0), Xs.std(axis=0) + EPS

def log_lr(x, mg, sg, mr, sr):
    return float(np.sum(stats.norm.logpdf(x, mg, sg) - stats.norm.logpdf(x, mr, sr)))

def norm01(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi - lo > EPS else np.zeros_like(v)

good_m = y_cls == "Good"
mg, sg = fit_gauss(X[good_m])
mr, sr = fit_gauss(X[~good_m])

lr_train_raw = np.array([log_lr(x, mg, sg, mr, sr) for x in X])
lr_train     = norm01(lr_train_raw)

print("  Running LOO...")
lr_loo_raw = np.zeros(len(X))
for tr, te in LeaveOneOut().split(X):
    gm = y_cls[tr] == "Good"
    rm = ~gm
    if gm.sum() >= 2 and rm.sum() >= 2:
        mg2,sg2 = fit_gauss(X[tr][gm])
        mr2,sr2 = fit_gauss(X[tr][rm])
        lr_loo_raw[te[0]] = log_lr(X[te[0]], mg2, sg2, mr2, sr2)

lr_loo = norm01(lr_loo_raw)

r_lr_tr, _ = pearsonr(lr_train, y_llm)
r_lr_lo, _ = pearsonr(lr_loo,   y_llm)
print(f"  Pearson r (train) = {r_lr_tr:.4f}")
print(f"  Pearson r (LOO)   = {r_lr_lo:.4f}")


# ============================================================
#  FINAL COMPARISON TABLE
# ============================================================
r_qimg,  _ = pearsonr(df["q_image"].values,     y_llm)
r_tsas,  _ = pearsonr(df["tsas"].values,         y_llm)
r_wtsas, _ = pearsonr(df["wtsas_final"].values,  y_llm)

print("\n" + "=" * 60)
print("  FINAL COMPARISON — Pearson r with LLM Ratings")
print("=" * 60)
print(f"\n  {'Metric':<35} {'r':>8}  {'Type'}")
print(f"  {'-'*55}")
print(f"  {'Q_image (CLIP score)':<35} {r_qimg:>8.4f}  best single feature")
print(f"  {'TSAS (ImageBind coherence)':<35} {r_tsas:>8.4f}  baseline coherence")
print(f"  {'WTSAS (quality-weighted)':<35} {r_wtsas:>8.4f}  your Stage 5")
print(f"  {'Naive Bayes P(Good) - train':<35} {r_nb_tr:>8.4f}  overfit estimate")
print(f"  {'Naive Bayes P(Good) - LOO':<35} {r_nb_lo:>8.4f}  honest NB score")
print(f"  {'LR Good-vs-Rest - train':<35} {r_lr_tr:>8.4f}  overfit estimate")
print(f"  {'LR Good-vs-Rest - LOO':<35} {r_lr_lo:>8.4f}  honest LR score")


# ============================================================
#  SAVE
# ============================================================
df_out = df.copy()
df_out["p_good_train"] = np.round(p_train, 4)
df_out["p_good_loo"]   = np.round(p_loo,   4)
df_out["lr_gr_train"]  = np.round(lr_train, 4)
df_out["lr_gr_loo"]    = np.round(lr_loo,   4)

nb_path = SCORES / "naive_bayes_results.csv"
lr_path = SCORES / "likelihood_ratio_results.csv"

df_out[list(df.columns) + ["p_good_train","p_good_loo"]].to_csv(nb_path, index=False)
df_out[list(df.columns) + ["lr_gr_train","lr_gr_loo"]].to_csv(lr_path,   index=False)

# comparison summary CSV
summary = pd.DataFrame([
    {"metric": "Q_image (CLIP)",         "pearson_r": round(r_qimg,  4), "type": "single feature"},
    {"metric": "TSAS (ImageBind)",        "pearson_r": round(r_tsas,  4), "type": "coherence baseline"},
    {"metric": "WTSAS (final)",           "pearson_r": round(r_wtsas, 4), "type": "Stage 5"},
    {"metric": "Naive Bayes LOO",         "pearson_r": round(r_nb_lo, 4), "type": "probabilistic"},
    {"metric": "Likelihood Ratio LOO",    "pearson_r": round(r_lr_lo, 4), "type": "probabilistic"},
])
summary.to_csv(SCORES / "final_comparison.csv", index=False)

print(f"\n  Saved:")
print(f"    {nb_path}")
print(f"    {lr_path}")
print(f"    {SCORES/'final_comparison.csv'}")
print(f"\n  ALL DONE.")
