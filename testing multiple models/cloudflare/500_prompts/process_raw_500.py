"""
process_raw_500.py
==================
Reads  : raw_scores_500.csv  — teammate's raw features for all 500 prompts
         quality_scores.csv  — 53-prompt quality features (our pipeline)
         llm_ratings.csv     — 53-prompt Gemini ratings (ground-truth labels)

Computes Q_text, Q_image, Q_audio for 500 prompts.
Trains Naive Bayes + Likelihood Ratio on the 53 labeled prompts
(with z-score normalization for cross-dataset compatibility),
then scores all 500 prompts.

Output:
  scores/final_scores_500.csv   — all 500 prompts with derived metrics
  scores/comparison_summary.csv — Pearson r table (53-prompt LOO estimates)

NOTE: wer_inv in raw_scores_500.csv is computed against the short prompt text
(TTS faithfully reads prompt → high wer_inv ≈ 0.88), while our 53-prompt
pipeline computed wer_inv against the longer generated text (→ low ≈ 0.17).
q_audio therefore has a different baseline; z-score normalization corrects for
this so the NB/LR models remain applicable across datasets.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut

# ── paths ────────────────────────────────────────────────────────────────────
HERE     = Path(__file__).parent
RAW_500  = Path(r"C:\Users\hp\OneDrive\Desktop\raw_scores_500.csv")
QS_53    = Path(r"D:\nihal\Collaborative-Project\testing multiple models"
                r"\cloudflare\outputs\scores\quality_scores.csv")
LLM_53   = Path(r"D:\nihal\Collaborative-Project\testing multiple models"
                r"\cloudflare\outputs\scores\llm_ratings.csv")
OUT_CSV  = HERE / "scores" / "final_scores_500.csv"
SUM_CSV  = HERE / "scores" / "comparison_summary.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

LAMBDA = 0.5   # variance penalty in quality_pair
FEATURES = ["q_text", "q_image", "q_audio"]

# ── quality_pair formula ─────────────────────────────────────────────────────
def quality_pair(a, b, lam=LAMBDA):
    avg = (a + b) / 2.0
    var = ((a - avg) ** 2 + (b - avg) ** 2) / 2.0
    return avg - lam * var


# ── 1. Load and compute quality features for 500 prompts ────────────────────
print("=" * 65)
print("  Loading raw_scores_500.csv  (500 prompts, no LLM labels)")
print("=" * 65)
raw = pd.read_csv(RAW_500)
print(f"  Rows: {len(raw)}  Columns: {list(raw.columns)}")

raw["q_text"]  = raw["bertscore"]
raw["q_image"] = quality_pair(raw["clip"],     raw["aesthetic"])
raw["q_audio"] = quality_pair(raw["semantic"], raw["wer_inv"])

print(f"\n  Quality feature statistics (500 prompts):")
for c in FEATURES:
    print(f"    {c:10s}: mean={raw[c].mean():.4f}  std={raw[c].std():.4f}  "
          f"min={raw[c].min():.4f}  max={raw[c].max():.4f}")


# ── 2. Load 53-prompt reference data (quality + LLM labels) ─────────────────
print("\n" + "=" * 65)
print("  Loading 53-prompt labeled data (quality_scores + llm_ratings)")
print("=" * 65)
qs53  = pd.read_csv(QS_53)
llm53 = pd.read_csv(LLM_53)
ref   = pd.merge(qs53, llm53[["prompt_id", "llm_overall",
                               "llm_text", "llm_image", "llm_audio"]],
                 on="prompt_id")
print(f"  Merged: {len(ref)} rows")

X53  = ref[FEATURES].values
y53  = ref["llm_overall"].values
X500 = raw[FEATURES].values


# ── 3. Z-score normalization (fit on 53, transform both) ────────────────────
mu  = X53.mean(axis=0)
sig = X53.std(axis=0) + 1e-9
X53_z  = (X53  - mu) / sig
X500_z = (X500 - mu) / sig

print(f"\n  Z-score normalization fitted on 53-prompt data:")
for i, f in enumerate(FEATURES):
    print(f"    {f:10s}: mu={mu[i]:.4f}  sigma={sig[i]:.4f}")


# ── 4. Label binning ─────────────────────────────────────────────────────────
def bin_score(s):
    if s >= 8:   return "Good"
    if s >= 6:   return "Medium"
    return "Bad"

y_cls53 = np.array([bin_score(v) for v in y53])
print(f"\n  Class distribution (53 prompts):")
for cls in ["Bad", "Medium", "Good"]:
    print(f"    {cls:7s}: {(y_cls53 == cls).sum()}")


# ═══════════════════════════════════════════════════════════════
#  NAIVE BAYES
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("  NAIVE BAYES  (train on 53, LOO validation, predict 500)")
print("─" * 65)

gnb  = GaussianNB().fit(X53_z, y_cls53)
cidx = list(gnb.classes_).index("Good")

# LOO on 53-prompt set
p_loo_53 = np.zeros(len(X53_z))
for tr, te in LeaveOneOut().split(X53_z):
    g  = GaussianNB().fit(X53_z[tr], y_cls53[tr])
    cl = list(g.classes_)
    p_loo_53[te[0]] = (g.predict_proba(X53_z[te])[0, cl.index("Good")]
                       if "Good" in cl else 0.0)

p_train_53 = gnb.predict_proba(X53_z)[:, cidx]
r_nb_tr, _ = pearsonr(p_train_53, y53)
r_nb_lo, _ = pearsonr(p_loo_53,   y53)
print(f"  Pearson r (train-53) = {r_nb_tr:.4f}")
print(f"  Pearson r (LOO-53)   = {r_nb_lo:.4f}   ← honest estimate")

# Predict 500 prompts
raw["nb_p_good"] = np.round(gnb.predict_proba(X500_z)[:, cidx], 4)


# ═══════════════════════════════════════════════════════════════
#  LIKELIHOOD RATIO (Good-vs-Rest)
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("  LIKELIHOOD RATIO — Good-vs-Rest  (train on 53, apply to 500)")
print("─" * 65)

EPS = 1e-6

def fit_gauss(Xs):
    return Xs.mean(axis=0), Xs.std(axis=0) + EPS

def log_lr(x, mg, sg, mr, sr):
    return float(np.sum(stats.norm.logpdf(x, mg, sg)
                        - stats.norm.logpdf(x, mr, sr)))

def norm01(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi - lo > EPS else np.zeros_like(v)

good_m = y_cls53 == "Good"
mg53, sg53 = fit_gauss(X53_z[good_m])
mr53, sr53 = fit_gauss(X53_z[~good_m])

# LOO on 53-prompt set
lr_loo_raw = np.zeros(len(X53_z))
for tr, te in LeaveOneOut().split(X53_z):
    gm = y_cls53[tr] == "Good"
    rm = ~gm
    if gm.sum() >= 2 and rm.sum() >= 2:
        mg2, sg2 = fit_gauss(X53_z[tr][gm])
        mr2, sr2 = fit_gauss(X53_z[tr][rm])
        lr_loo_raw[te[0]] = log_lr(X53_z[te[0]], mg2, sg2, mr2, sr2)

lr_raw_53   = np.array([log_lr(x, mg53, sg53, mr53, sr53) for x in X53_z])
lr_train_53 = norm01(lr_raw_53)
lr_loo_53   = norm01(lr_loo_raw)

r_lr_tr, _ = pearsonr(lr_train_53, y53)
r_lr_lo, _ = pearsonr(lr_loo_53,   y53)
print(f"  Pearson r (train-53) = {r_lr_tr:.4f}")
print(f"  Pearson r (LOO-53)   = {r_lr_lo:.4f}   ← honest estimate")

# Predict 500 prompts
lr_raw_500 = np.array([log_lr(x, mg53, sg53, mr53, sr53) for x in X500_z])
raw["lr_gr"] = np.round(norm01(lr_raw_500), 4)


# ═══════════════════════════════════════════════════════════════
#  QUALITY-WEIGHTED COMPOSITE (without ImageBind)
# ═══════════════════════════════════════════════════════════════
# Weights learned via linear regression on 53-prompt data
W1, W2, W3 = 0.287, 0.520, 0.193
W_SUM = W1 + W2 + W3
raw["q_combined"] = np.round(
    (W1 * raw["q_text"] + W2 * raw["q_image"] + W3 * raw["q_audio"]) / W_SUM, 4)


# ── 53-prompt reference correlations ─────────────────────────────────────────
r_qtxt, _ = pearsonr(ref["q_text"].values,  y53)
r_qimg, _ = pearsonr(ref["q_image"].values, y53)
r_qaud, _ = pearsonr(ref["q_audio"].values, y53)
q_comb53  = (W1 * ref["q_text"] + W2 * ref["q_image"] + W3 * ref["q_audio"]) / W_SUM
r_qcmb, _ = pearsonr(q_comb53.values, y53)

# TSAS and WTSAS from the original wtsas_final.csv (if available)
wts_path = Path(r"D:\nihal\Collaborative-Project\testing multiple models"
                r"\cloudflare\outputs\scores\wtsas_final.csv")
if wts_path.exists():
    wts = pd.read_csv(wts_path)
    w_ref = pd.merge(ref[["prompt_id"]], wts, on="prompt_id")
    y_w   = ref.set_index("prompt_id").loc[w_ref["prompt_id"], "llm_overall"].values
    r_tsas,  _ = pearsonr(w_ref["tsas"].values,       y_w)
    r_wtsas, _ = pearsonr(w_ref["wtsas_final"].values, y_w)
else:
    r_tsas, r_wtsas = float("nan"), float("nan")


# ═══════════════════════════════════════════════════════════════
#  FINAL COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  COMPARISON — Pearson r vs LLM ratings  (53-prompt LOO)")
print("=" * 65)

rows = [
    ("Q_text (BERTScore)",        r_qtxt,   "single feature"),
    ("Q_image (CLIP+aesthetic)",   r_qimg,   "single feature — best"),
    ("Q_audio (semantic+wer_inv)", r_qaud,   "single feature"),
    ("Q_combined (w-avg quality)", r_qcmb,   "quality composite"),
    ("TSAS (ImageBind coherence)", r_tsas,   "coherence baseline"),
    ("WTSAS (quality-weighted)",   r_wtsas,  "main metric"),
    ("Naive Bayes P(Good) — LOO",  r_nb_lo,  "probabilistic (honest)"),
    ("LR Good-vs-Rest — LOO",      r_lr_lo,  "probabilistic (honest)"),
]

print(f"\n  {'Metric':<37} {'r':>8}  {'Notes'}")
print(f"  {'─'*70}")
for name, r, note in rows:
    flag = " ←" if abs(r) == max(abs(row[1]) for row in rows
                                 if not np.isnan(row[1])) else ""
    print(f"  {name:<37} {r:>8.4f}  {note}{flag}")


# ── 500-prompt distribution ───────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  500-PROMPT SCORE DISTRIBUTIONS  (no LLM labels)")
print("=" * 65)
for c in ["q_text", "q_image", "q_audio", "q_combined", "nb_p_good", "lr_gr"]:
    s = raw[c]
    print(f"  {c:15s}: mean={s.mean():.4f}  std={s.std():.4f}  "
          f"min={s.min():.4f}  max={s.max():.4f}")

# Top 10 / Bottom 10 by NB score
print("\n  Top-10 prompts by nb_p_good:")
top10 = raw.nlargest(10, "nb_p_good")[["prompt_id","prompt","nb_p_good","q_image","q_text"]]
print(top10.to_string(index=False))

print("\n  Bottom-10 prompts by nb_p_good:")
bot10 = raw.nsmallest(10, "nb_p_good")[["prompt_id","prompt","nb_p_good","q_image","q_text"]]
print(bot10.to_string(index=False))


# ═══════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════
out_cols = ["prompt_id", "prompt",
            "q_text", "q_image", "q_audio", "q_combined",
            "nb_p_good", "lr_gr"]
raw[out_cols].to_csv(OUT_CSV, index=False)
print(f"\n  Saved: {OUT_CSV}")

summary = pd.DataFrame([
    {"metric": n, "pearson_r_vs_llm_53": round(r, 4) if not np.isnan(r) else "n/a",
     "notes": t}
    for n, r, t in rows
])
summary.to_csv(SUM_CSV, index=False)
print(f"  Saved: {SUM_CSV}")

print(f"\n  ALL DONE.  {len(raw)} prompts scored.")
