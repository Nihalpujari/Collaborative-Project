"""
run_all_500.py
==============
Now that we have LLM judge ratings for all 500 prompts, run all 3 approaches:

  Approach 1 — Linear Regression  : learn w1,w2,w3 → Q_combined score
  Approach 2 — Naive Bayes        : P(Good) with LOO CV
  Approach 3 — Likelihood Ratio   : log P(x|Good) - log P(x|Rest), LOO CV

Inputs:
  raw_scores_500.csv               → Q_text, Q_image, Q_audio for 500 prompts
  judge_gemini_gemini-3.1-flash-lite.csv → LLM judge ratings (judge_mean, 1-5 scale)

Output:
  scores/all_500_results.csv       → all scores for all 500 prompts
  scores/all_500_summary.csv       → Pearson r comparison table
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

HERE    = Path(__file__).parent
RAW     = Path(r"C:\Users\hp\OneDrive\Desktop\raw_scores_500.csv")
JUDGE   = Path(r"C:\Users\hp\OneDrive\Desktop\judge_gemini_gemini-3.1-flash-lite.csv")
OUT_CSV = HERE / "scores" / "all_500_results.csv"
SUM_CSV = HERE / "scores" / "all_500_summary.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

LAMBDA   = 0.5
FEATURES = ["q_text", "q_image", "q_audio"]

# ── quality_pair ─────────────────────────────────────────────────────────────
def quality_pair(a, b, lam=LAMBDA):
    avg = (a + b) / 2.0
    var = ((a - avg) ** 2 + (b - avg) ** 2) / 2.0
    return avg - lam * var

# ── 1. Compute Q features ────────────────────────────────────────────────────
print("=" * 65)
print("  Loading data")
print("=" * 65)

raw   = pd.read_csv(RAW)
judge = pd.read_csv(JUDGE)

raw["q_text"]  = raw["bertscore"]
raw["q_image"] = quality_pair(raw["clip"],     raw["aesthetic"])
raw["q_audio"] = quality_pair(raw["semantic"], raw["wer_inv"])

# ── 2. Merge on prompt_id ────────────────────────────────────────────────────
df = pd.merge(raw, judge[["prompt_id","judge_mean","judge_grade",
                           "text_score","image_score"]], on="prompt_id")
print(f"  Merged: {len(df)} prompts with Q features + LLM ratings")

X = df[FEATURES].values
y = df["judge_mean"].values      # ground truth: 1-5 scale

print(f"\n  judge_mean stats: mean={y.mean():.3f}  std={y.std():.3f}  "
      f"min={y.min():.2f}  max={y.max():.2f}")

# ── Label binning ─────────────────────────────────────────────────────────────
def bin_score(s):
    if s >= 4.0:  return "Good"
    if s >= 3.0:  return "Medium"
    return "Bad"

y_cls = np.array([bin_score(v) for v in y])
print(f"\n  Class distribution (500 prompts):")
for cls in ["Bad", "Medium", "Good"]:
    print(f"    {cls:7s}: {(y_cls == cls).sum()}")


# ═══════════════════════════════════════════════════════════════
#  APPROACH 1 — LINEAR REGRESSION  (learn weights from 500)
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 65)
print("  APPROACH 1 — LINEAR REGRESSION")
print("  X = [Q_text, Q_image, Q_audio]   →   y = judge_mean")
print("═" * 65)

lr_model = LinearRegression().fit(X, y)
w1, w2, w3 = lr_model.coef_
bias       = lr_model.intercept_
R2         = lr_model.score(X, y)

print(f"\n  Learned weights:")
print(f"    w1 (Q_text)  = {w1:.4f}")
print(f"    w2 (Q_image) = {w2:.4f}")
print(f"    w3 (Q_audio) = {w3:.4f}")
print(f"    bias         = {bias:.4f}")
print(f"    R²           = {R2:.4f}")

# LOO prediction for honest estimate
lr_loo_pred = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
r_lr_tr, _ = pearsonr(lr_model.predict(X), y)
r_lr_lo, _ = pearsonr(lr_loo_pred, y)
print(f"\n  Pearson r (train) = {r_lr_tr:.4f}")
print(f"  Pearson r (LOO)   = {r_lr_lo:.4f}   <- honest estimate")

# Q_combined using new weights
W_SUM = w1 + w2 + w3
df["q_combined"] = np.round(
    (w1 * df["q_text"] + w2 * df["q_image"] + w3 * df["q_audio"]) / W_SUM, 4)
df["lr_pred"]    = np.round(lr_loo_pred, 4)

# Individual feature correlations
r_qt, _ = pearsonr(df["q_text"].values,  y)
r_qi, _ = pearsonr(df["q_image"].values, y)
r_qa, _ = pearsonr(df["q_audio"].values, y)
r_qc, _ = pearsonr(df["q_combined"].values, y)
print(f"\n  Individual feature correlations:")
print(f"    Q_text  r = {r_qt:.4f}")
print(f"    Q_image r = {r_qi:.4f}")
print(f"    Q_audio r = {r_qa:.4f}")
print(f"    Q_combined (weighted) r = {r_qc:.4f}")


# ═══════════════════════════════════════════════════════════════
#  APPROACH 2 — NAIVE BAYES  (LOO on 500)
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 65)
print("  APPROACH 2 — NAIVE BAYES  (LOO CV on 500 prompts)")
print("═" * 65)

gnb  = GaussianNB().fit(X, y_cls)
cidx = list(gnb.classes_).index("Good")

p_train = gnb.predict_proba(X)[:, cidx]

print("  Running LOO (500 iterations)...")
p_loo = np.zeros(len(X))
for tr, te in LeaveOneOut().split(X):
    g  = GaussianNB().fit(X[tr], y_cls[tr])
    cl = list(g.classes_)
    p_loo[te[0]] = (g.predict_proba(X[te])[0, cl.index("Good")]
                    if "Good" in cl else 0.0)

r_nb_tr, _ = pearsonr(p_train, y)
r_nb_lo, _ = pearsonr(p_loo,   y)
print(f"  Pearson r (train) = {r_nb_tr:.4f}")
print(f"  Pearson r (LOO)   = {r_nb_lo:.4f}   <- honest estimate")

df["nb_p_good"]     = np.round(p_train, 4)
df["nb_p_good_loo"] = np.round(p_loo,   4)


# ═══════════════════════════════════════════════════════════════
#  APPROACH 3 — LIKELIHOOD RATIO  (LOO on 500)
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 65)
print("  APPROACH 3 — LIKELIHOOD RATIO  Good-vs-Rest  (LOO on 500)")
print("═" * 65)

EPS = 1e-6

def fit_gauss(Xs):
    return Xs.mean(axis=0), Xs.std(axis=0) + EPS

def log_lr(x, mg, sg, mr, sr):
    return float(np.sum(stats.norm.logpdf(x, mg, sg)
                        - stats.norm.logpdf(x, mr, sr)))

def norm01(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi - lo > EPS else np.zeros_like(v)

good_m = y_cls == "Good"
mg, sg = fit_gauss(X[good_m])
mr, sr = fit_gauss(X[~good_m])

lr_raw_train = np.array([log_lr(x, mg, sg, mr, sr) for x in X])
lr_train     = norm01(lr_raw_train)

print("  Running LOO (500 iterations)...")
lr_loo_raw = np.zeros(len(X))
for tr, te in LeaveOneOut().split(X):
    gm = y_cls[tr] == "Good"
    rm = ~gm
    if gm.sum() >= 2 and rm.sum() >= 2:
        mg2, sg2 = fit_gauss(X[tr][gm])
        mr2, sr2 = fit_gauss(X[tr][rm])
        lr_loo_raw[te[0]] = log_lr(X[te[0]], mg2, sg2, mr2, sr2)

lr_loo = norm01(lr_loo_raw)

r_lrg_tr, _ = pearsonr(lr_train, y)
r_lrg_lo, _ = pearsonr(lr_loo,   y)
print(f"  Pearson r (train) = {r_lrg_tr:.4f}")
print(f"  Pearson r (LOO)   = {r_lrg_lo:.4f}   <- honest estimate")

df["lr_gr"]     = np.round(lr_train, 4)
df["lr_gr_loo"] = np.round(lr_loo,   4)


# ═══════════════════════════════════════════════════════════════
#  FINAL COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 65)
print("  FINAL COMPARISON — Pearson r vs judge_mean (500 prompts)")
print("═" * 65)

rows = [
    ("Q_text (BERTScore)",           r_qt,     "single feature"),
    ("Q_image (CLIP+aesthetic)",      r_qi,     "single feature"),
    ("Q_audio (semantic+wer_inv)",    r_qa,     "single feature"),
    ("Q_combined (Linear Reg LOO)",   r_lr_lo,  "Approach 1 - honest"),
    ("Naive Bayes P(Good) - LOO",     r_nb_lo,  "Approach 2 - honest"),
    ("LR Good-vs-Rest - LOO",         r_lrg_lo, "Approach 3 - honest"),
]

print(f"\n  {'Metric':<38} {'r':>8}  {'Notes'}")
print(f"  {'─'*70}")
best_r = max(abs(r) for _, r, _ in rows)
for name, r, note in rows:
    flag = "  <- BEST" if abs(r) == best_r else ""
    print(f"  {name:<38} {r:>8.4f}  {note}{flag}")


# ═══════════════════════════════════════════════════════════════
#  TOP / BOTTOM 10
# ═══════════════════════════════════════════════════════════════
print("\n  Top-10 prompts by nb_p_good_loo:")
top = df.nlargest(10, "nb_p_good_loo")[["prompt_id","prompt","judge_mean","nb_p_good_loo","q_image"]]
print(top.to_string(index=False))

print("\n  Bottom-10 prompts by nb_p_good_loo:")
bot = df.nsmallest(10, "nb_p_good_loo")[["prompt_id","prompt","judge_mean","nb_p_good_loo","q_image"]]
print(bot.to_string(index=False))


# ═══════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════
out_cols = ["prompt_id", "prompt",
            "q_text", "q_image", "q_audio", "q_combined",
            "judge_mean", "judge_grade", "text_score", "image_score",
            "lr_pred",
            "nb_p_good", "nb_p_good_loo",
            "lr_gr", "lr_gr_loo"]
df[out_cols].to_csv(OUT_CSV, index=False)
print(f"\n  Saved: {OUT_CSV}")

summary = pd.DataFrame([
    {"metric": n, "pearson_r": round(r, 4), "notes": t}
    for n, r, t in rows
])
summary.to_csv(SUM_CSV, index=False)
print(f"  Saved: {SUM_CSV}")

print(f"\n  ALL DONE — {len(df)} prompts, 3 approaches complete.")
print(f"\n  New weights learned from 500 prompts:")
print(f"    w1={w1:.4f}  w2={w2:.4f}  w3={w3:.4f}  R2={R2:.4f}")
