"""
run_all_500_v2.py
=================
Updated pipeline — NB and LR now use [s1_weighted, s2_weighted, s3_weighted]
as features instead of raw [Q_text, Q_image, Q_audio].

Steps:
  1. Linear Regression: [Q_text, Q_image, Q_audio] → judge_mean → learns w1,w2,w3
  2. Compute s1_w, s2_w, s3_w using learned weights + ImageBind scores
  3. Approach 1 — WTSAS:  unchanged (from wtsas_500.csv)
  4. Approach 2 — NB:     GaussianNB on [s1_w, s2_w, s3_w], LOO CV
  5. Approach 3 — LR:     log P(x|Good) - log P(x|Rest) on [s1_w, s2_w, s3_w], LOO CV

Output:
  scores/all_500_v2_results.csv
  scores/all_500_v2_summary.csv
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
IB_CSV  = HERE / "scores" / "imagebind_500.csv"
WTSAS   = HERE / "scores" / "wtsas_500.csv"
OUT_CSV = HERE / "scores" / "all_500_v2_results.csv"
SUM_CSV = HERE / "scores" / "all_500_v2_summary.csv"

LAMBDA = 0.5

def quality_pair(a, b, lam=LAMBDA):
    avg = (a + b) / 2.0
    var = ((a - avg)**2 + (b - avg)**2) / 2.0
    return avg - lam * var

# ── 1. Load & compute Q features ─────────────────────────────────────────────
print("=" * 65)
print("  Loading data")
print("=" * 65)

raw   = pd.read_csv(RAW)
judge = pd.read_csv(JUDGE)
ib    = pd.read_csv(IB_CSV)

raw["q_text"]  = raw["bertscore"]
raw["q_image"] = quality_pair(raw["clip"],     raw["aesthetic"])
raw["q_audio"] = quality_pair(raw["semantic"], raw["wer_inv"])

df = pd.merge(raw, judge[["prompt_id","judge_mean","judge_grade"]], on="prompt_id")
df = pd.merge(df,  ib[["prompt_id","s1_text_image","s2_text_audio","s3_image_audio"]], on="prompt_id")

print(f"  Prompts with Q + judge + ImageBind scores: {len(df)}")

y    = df["judge_mean"].values
X_q  = df[["q_text","q_image","q_audio"]].values   # quality features

print(f"  judge_mean: mean={y.mean():.3f}  std={y.std():.3f}")

# ── 2. Linear Regression → learn w1, w2, w3 ─────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 1 — Linear Regression on Q features → learn weights")
print("=" * 65)

lr_model = LinearRegression().fit(X_q, y)
w1, w2, w3 = lr_model.coef_
bias = lr_model.intercept_
R2   = lr_model.score(X_q, y)

print(f"  w1 (Q_text)  = {w1:.4f}")
print(f"  w2 (Q_image) = {w2:.4f}")
print(f"  w3 (Q_audio) = {w3:.4f}")
print(f"  bias         = {bias:.4f}")
print(f"  R²           = {R2:.4f}")

# ── 3. Compute s1_w, s2_w, s3_w ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 2 — Compute quality-weighted coherence scores")
print("=" * 65)

qt = df["q_text"].values
qi = df["q_image"].values
qa = df["q_audio"].values
s1 = df["s1_text_image"].values
s2 = df["s2_text_audio"].values
s3 = df["s3_image_audio"].values

s1_w = s1 * (w1*qt + w2*qi) / (w1 + w2)
s2_w = s2 * (w1*qt + w3*qa) / (w1 + w3)
s3_w = s3 * (w2*qi + w3*qa) / (w2 + w3)

df["s1_weighted"] = np.round(s1_w, 4)
df["s2_weighted"] = np.round(s2_w, 4)
df["s3_weighted"] = np.round(s3_w, 4)

X_sw = np.column_stack([s1_w, s2_w, s3_w])   # NEW feature matrix

r_s1w, _ = pearsonr(s1_w, y)
r_s2w, _ = pearsonr(s2_w, y)
r_s3w, _ = pearsonr(s3_w, y)
print(f"  s1_weighted r = {r_s1w:.4f}")
print(f"  s2_weighted r = {r_s2w:.4f}")
print(f"  s3_weighted r = {r_s3w:.4f}")

# ── 4. Label binning ─────────────────────────────────────────────────────────
def bin_score(s):
    if s >= 4.0: return "Good"
    if s >= 3.0: return "Medium"
    return "Bad"

y_cls = np.array([bin_score(v) for v in y])
print(f"\n  Class distribution:")
for cls in ["Bad","Medium","Good"]:
    print(f"    {cls:7s}: {(y_cls==cls).sum()}")

# ── 5. Approach 2 — Naive Bayes on [s1_w, s2_w, s3_w] ───────────────────────
print("\n" + "=" * 65)
print("  APPROACH 2 — Naive Bayes on [s1_w, s2_w, s3_w]  (LOO CV)")
print("=" * 65)

gnb  = GaussianNB().fit(X_sw, y_cls)
cidx = list(gnb.classes_).index("Good")

p_train = gnb.predict_proba(X_sw)[:, cidx]

print("  Running LOO (500 iterations)...")
p_loo = np.zeros(len(X_sw))
for tr, te in LeaveOneOut().split(X_sw):
    g  = GaussianNB().fit(X_sw[tr], y_cls[tr])
    cl = list(g.classes_)
    p_loo[te[0]] = (g.predict_proba(X_sw[te])[0, cl.index("Good")]
                    if "Good" in cl else 0.0)

r_nb_tr, _ = pearsonr(p_train, y)
r_nb_lo, _ = pearsonr(p_loo,   y)
print(f"  Pearson r (train) = {r_nb_tr:.4f}")
print(f"  Pearson r (LOO)   = {r_nb_lo:.4f}   <- honest estimate")

df["nb_sw_train"] = np.round(p_train, 4)
df["nb_sw_loo"]   = np.round(p_loo,   4)

# ── 6. Approach 3 — LR Good-vs-Rest on [s1_w, s2_w, s3_w] ──────────────────
print("\n" + "=" * 65)
print("  APPROACH 3 — Likelihood Ratio on [s1_w, s2_w, s3_w]  (LOO CV)")
print("=" * 65)

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
mg, sg = fit_gauss(X_sw[good_m])
mr, sr = fit_gauss(X_sw[~good_m])

lr_raw_train = np.array([log_lr(x, mg, sg, mr, sr) for x in X_sw])
lr_train     = norm01(lr_raw_train)

print("  Running LOO (500 iterations)...")
lr_loo_raw = np.zeros(len(X_sw))
for tr, te in LeaveOneOut().split(X_sw):
    gm = y_cls[tr] == "Good"
    rm = ~gm
    if gm.sum() >= 2 and rm.sum() >= 2:
        mg2, sg2 = fit_gauss(X_sw[tr][gm])
        mr2, sr2 = fit_gauss(X_sw[tr][rm])
        lr_loo_raw[te[0]] = log_lr(X_sw[te[0]], mg2, sg2, mr2, sr2)

lr_loo = norm01(lr_loo_raw)

r_lr_tr, _ = pearsonr(lr_train, y)
r_lr_lo, _ = pearsonr(lr_loo,   y)
print(f"  Pearson r (train) = {r_lr_tr:.4f}")
print(f"  Pearson r (LOO)   = {r_lr_lo:.4f}   <- honest estimate")

df["lr_sw_train"] = np.round(lr_train, 4)
df["lr_sw_loo"]   = np.round(lr_loo,   4)

# ── 7. Load WTSAS (Approach 1 — unchanged) ───────────────────────────────────
wtsas_df = pd.read_csv(WTSAS)[["prompt_id","tsas","wtsas_final"]]
df = pd.merge(df, wtsas_df, on="prompt_id", how="left")

r_wtsas, _ = pearsonr(df["wtsas_final"].values, y)

# ── 8. Final comparison ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FINAL COMPARISON — Pearson r vs judge_mean (500 prompts)")
print("=" * 65)

r_qt, _ = pearsonr(qt, y)
r_qi, _ = pearsonr(qi, y)
r_qa, _ = pearsonr(qa, y)
r_s1, _ = pearsonr(s1, y)
r_s2, _ = pearsonr(s2, y)
r_s3, _ = pearsonr(s3, y)

rows = [
    ("Q_text  (BERTScore)",              r_qt,    "quality feature"),
    ("Q_image (CLIP+aesthetic)",          r_qi,    "quality feature"),
    ("Q_audio (semantic+wer_inv)",        r_qa,    "quality feature"),
    ("s1  raw (text↔image)",             r_s1,    "coherence feature"),
    ("s2  raw (text↔audio)",             r_s2,    "coherence feature"),
    ("s3  raw (image↔audio)",            r_s3,    "coherence feature"),
    ("s1_weighted",                       r_s1w,   "quality × coherence"),
    ("s2_weighted",                       r_s2w,   "quality × coherence"),
    ("s3_weighted",                       r_s3w,   "quality × coherence"),
    ("WTSAS final  [Approach 1]",        r_wtsas, "coherence metric"),
    ("NB on [s1_w,s2_w,s3_w] LOO  [Approach 2]", r_nb_lo, "← NEW"),
    ("LR on [s1_w,s2_w,s3_w] LOO  [Approach 3]", r_lr_lo, "← NEW"),
]

print(f"\n  {'Metric':<45} {'r':>8}  Notes")
print(f"  {'─'*68}")
best = max(abs(r) for _,r,_ in rows)
for name, r, note in rows:
    flag = "  *** BEST" if abs(r) == best else ""
    print(f"  {name:<45} {r:>8.4f}  {note}{flag}")

# ── 9. Save ───────────────────────────────────────────────────────────────────
out_cols = ["prompt_id","prompt","q_text","q_image","q_audio",
            "s1_text_image","s2_text_audio","s3_image_audio",
            "s1_weighted","s2_weighted","s3_weighted",
            "judge_mean","judge_grade",
            "tsas","wtsas_final",
            "nb_sw_train","nb_sw_loo",
            "lr_sw_train","lr_sw_loo"]
df[out_cols].to_csv(OUT_CSV, index=False)
print(f"\n  Saved: {OUT_CSV}")

summary = pd.DataFrame([{"metric":n,"pearson_r":round(r,4),"notes":t} for n,r,t in rows])
summary.to_csv(SUM_CSV, index=False)
print(f"  Saved: {SUM_CSV}")
print(f"\n  ALL DONE — {len(df)} prompts.")
