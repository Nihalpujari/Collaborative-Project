"""
Stage 3 - Learn Weights via Linear Regression

Uses LLM judge ratings (llm_overall) as the target (y) and
[Q_text, Q_image, Q_audio] as features (X) to learn weights w1, w2, w3.

These weights tell us how much each modality contributes to overall quality
as judged by the LLM — replacing our hand-crafted equal weights.

Also finds optimal lambda for Q_image and Q_audio by maximizing correlation
with LLM overall score.

Saves: outputs/scores/learned_weights.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score

# -- paths -------------------------------------------------------------------
BASE        = Path(__file__).parent.parent
SCORES_CSV  = BASE / "outputs/scores/quality_scores.csv"
RATINGS_CSV = BASE / "outputs/scores/llm_ratings.csv"
OUT_DIR     = BASE / "outputs/scores"

# -- load data ---------------------------------------------------------------
df_scores  = pd.read_csv(SCORES_CSV)
df_ratings = pd.read_csv(RATINGS_CSV)

# Merge on prompt_id
df = pd.merge(df_scores, df_ratings[["prompt_id", "llm_overall", "llm_text", "llm_image", "llm_audio"]],
              on="prompt_id")

print("=" * 70)
print("  STAGE 3 - Learn Weights via Linear Regression")
print(f"  Samples: {len(df)}")
print("=" * 70 + "\n")

# -- features and target -----------------------------------------------------
X = df[["q_text", "q_image", "q_audio"]].values
y = df["llm_overall"].values

# Normalize y to 0-1 scale (from 1-10)
y_norm = (y - 1) / 9

# -- Step 1: Pearson correlation of each feature with target -----------------
print("Step 1 — Pearson Correlation with LLM Overall Score")
print("-" * 50)
for col, name in [("q_text", "Q_text"), ("q_image", "Q_image"), ("q_audio", "Q_audio")]:
    r, p = pearsonr(df[col].values, y)
    print(f"  {name:<12}  r = {r:+.4f}   p = {p:.4f}  {'*significant*' if p < 0.05 else ''}")

print()

# -- Step 2: Linear Regression -----------------------------------------------
print("Step 2 — Linear Regression: y = w1*Q_text + w2*Q_image + w3*Q_audio + b")
print("-" * 50)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

reg = LinearRegression()
reg.fit(X_scaled, y)

w1, w2, w3 = reg.coef_
b = reg.intercept_
y_pred = reg.predict(X_scaled)

r2   = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"  w1 (Q_text)   = {w1:+.4f}")
print(f"  w2 (Q_image)  = {w2:+.4f}")
print(f"  w3 (Q_audio)  = {w3:+.4f}")
print(f"  bias          = {b:+.4f}")
print(f"  R²            = {r2:.4f}")
print(f"  RMSE          = {rmse:.4f}")
print()

# Normalize weights to sum to 1 (for interpretability)
w_abs = np.abs([w1, w2, w3])
w_sum = w_abs.sum()
w1_n, w2_n, w3_n = w_abs / w_sum

print("  Normalized weights (sum to 1):")
print(f"  w1 (Q_text)   = {w1_n:.4f}  ({w1_n*100:.1f}%)")
print(f"  w2 (Q_image)  = {w2_n:.4f}  ({w2_n*100:.1f}%)")
print(f"  w3 (Q_audio)  = {w3_n:.4f}  ({w3_n*100:.1f}%)")
print()

# -- Step 3: Leave-One-Out Cross Validation ----------------------------------
print("Step 3 — Leave-One-Out Cross Validation")
print("-" * 50)

loo = LeaveOneOut()
y_loo_pred = np.zeros(len(y))

for train_idx, test_idx in loo.split(X_scaled):
    reg_loo = LinearRegression()
    reg_loo.fit(X_scaled[train_idx], y[train_idx])
    y_loo_pred[test_idx] = reg_loo.predict(X_scaled[test_idx])

loo_rmse = np.sqrt(mean_squared_error(y, y_loo_pred))
loo_r2   = r2_score(y, y_loo_pred)
loo_r, _ = pearsonr(y, y_loo_pred)

print(f"  LOO RMSE      = {loo_rmse:.4f}")
print(f"  LOO R²        = {loo_r2:.4f}")
print(f"  LOO Pearson r = {loo_r:.4f}")
print()

# -- Step 4: Find optimal lambda ---------------------------------------------
print("Step 4 — Find Optimal Lambda (for Q_image and Q_audio)")
print("-" * 50)

def quality_pair(a, b, lam):
    avg = (a + b) / 2
    var = float(np.var([a, b]))
    return float(np.clip(avg - lam * var, 0, 1))

def recompute_scores(df, lam):
    q_img = df.apply(lambda r: quality_pair(r["clip_score"], r["aesthetic"], lam), axis=1)
    q_aud = df.apply(lambda r: quality_pair(r["semantic_score"], r["wer_inv"], lam), axis=1)
    return q_img.values, q_aud.values

lambdas = np.arange(0.0, 2.05, 0.05)
best_lam, best_corr = 0.5, -1

lam_results = []
for lam in lambdas:
    q_img, q_aud = recompute_scores(df, lam)
    # Combined feature matrix with current lambda
    X_lam = np.column_stack([df["q_text"].values, q_img, q_aud])
    reg_lam = LinearRegression().fit(X_lam, y)
    y_lam_pred = reg_lam.predict(X_lam)
    r_lam, _ = pearsonr(y, y_lam_pred)
    lam_results.append({"lambda": round(lam, 2), "pearson_r": round(r_lam, 4)})
    if r_lam > best_corr:
        best_corr = r_lam
        best_lam  = lam

print(f"  Optimal lambda = {best_lam:.2f}  (Pearson r = {best_corr:.4f})")
print(f"  Current lambda = 0.50  (Pearson r = {[r['pearson_r'] for r in lam_results if r['lambda'] == 0.5][0]:.4f})")
print()

# -- Step 5: Save results ----------------------------------------------------
print("Step 5 — Saving results")
print("-" * 50)

# Predictions vs actuals
df_out = df[["prompt_id", "prompt", "q_text", "q_image", "q_audio", "llm_overall"]].copy()
df_out["predicted_score"] = np.round(y_pred, 3)
df_out["error"]           = np.round(y_pred - y, 3)
out_path = OUT_DIR / "learned_weights.csv"
df_out.to_csv(out_path, index=False)
print(f"  Predictions saved to: {out_path}")

# Lambda sweep results
lam_df = pd.DataFrame(lam_results)
lam_out = OUT_DIR / "lambda_sweep.csv"
lam_df.to_csv(lam_out, index=False)
print(f"  Lambda sweep saved to: {lam_out}")

# Summary weights
weights_summary = {
    "w1_q_text":       round(w1_n, 4),
    "w2_q_image":      round(w2_n, 4),
    "w3_q_audio":      round(w3_n, 4),
    "r2_train":        round(r2,   4),
    "rmse_train":      round(rmse, 4),
    "loo_r2":          round(loo_r2,   4),
    "loo_rmse":        round(loo_rmse, 4),
    "optimal_lambda":  round(best_lam, 2),
}
pd.DataFrame([weights_summary]).to_csv(OUT_DIR / "weights_summary.csv", index=False)
print(f"  Weights summary saved to: {OUT_DIR / 'weights_summary.csv'}")

print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(f"\n  Learned weights (normalized):")
print(f"    w1 (Q_text)  = {w1_n:.4f}  ({w1_n*100:.1f}%)")
print(f"    w2 (Q_image) = {w2_n:.4f}  ({w2_n*100:.1f}%)")
print(f"    w3 (Q_audio) = {w3_n:.4f}  ({w3_n*100:.1f}%)")
print(f"\n  Model fit:  R² = {r2:.4f}   RMSE = {rmse:.4f}")
print(f"  LOO CV:     R² = {loo_r2:.4f}  RMSE = {loo_rmse:.4f}")
print(f"\n  Optimal lambda = {best_lam:.2f}")
print(f"\n  These weights will be used in Stage 4 to compute")
print(f"  Quality-Weighted Coherence (WTSAS) scores.")


if __name__ == "__main__":
    pass
