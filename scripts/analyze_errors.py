# analyze_errors.py
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from sklearn.calibration import calibration_curve
from app.data.loader import load_dataset
from app.data.split import split_df
from app.embeddings.embedder import Embedder
from app.inference.predictor import Predictor
from app.models.explainer import Explainer

# ── Load ───────────────────────────────────────────────────────────────────────
predictor = Predictor()
print(f"Loaded threshold: {predictor.threshold:.4f}")

df = load_dataset("data/combined_data_final.csv")
train_df, test_df = split_df(df)

embedder = Embedder()
X_test   = np.array(embedder.encode(test_df["text"].astype(str).tolist()))
X_train  = np.array(embedder.encode(train_df["text"].astype(str).tolist()))
y_test   = test_df["label"].values
y_train  = train_df["label"].values

# ── Single predict_proba call — reused everywhere below ───────────────────────
# NOTE: we call predict_proba ONCE per split and store the result.
# Nothing here retrains or fits anything — y_test is only used to MEASURE,
# never to update the model. No cheating possible.
train_proba = predictor.model.predict_proba(X_train)[:, 1]   # raw scores, train
test_proba  = predictor.model.predict_proba(X_test)[:, 1]    # raw scores, test

train_pred  = (train_proba >= predictor.threshold).astype(int)
test_pred   = (test_proba  >= predictor.threshold).astype(int)

# ── Overfitting check ──────────────────────────────────────────────────────────
train_auc = roc_auc_score(y_train, train_proba)
test_auc  = roc_auc_score(y_test,  test_proba)
train_f1  = f1_score(y_train, train_pred)
test_f1   = f1_score(y_test,  test_pred)

print("\n=== OVERFITTING CHECK ===")
print(f"{'':15s} {'TRAIN':>10} {'TEST':>10} {'GAP':>10}")
print(f"{'AUC':15s} {train_auc:>10.4f} {test_auc:>10.4f} {train_auc - test_auc:>10.4f}")
print(f"{'F1 (scam)':15s} {train_f1:>10.4f} {test_f1:>10.4f} {train_f1 - test_f1:>10.4f}")
print(f"\nVerdict: {'healthy ✅' if train_auc - test_auc < 0.02 else 'OVERFITTING DETECTED 🚨'}")

# ── Calibration check ──────────────────────────────────────────────────────────
# We use test_proba (already computed above) and y_test (ground truth labels).
# calibration_curve() only READS these values to bin and count — no fitting,
# no model update. Completely safe.
#
# n_bins=8: our test set has only 137 scams, so 8 bins ≈ ~17 scams/bin.
# More bins would give too few samples per bucket and noisy results.
fraction_pos, mean_pred = calibration_curve(y_test, test_proba, n_bins=8, strategy="uniform")
brier = brier_score_loss(y_test, test_proba)

print(f"\n=== CALIBRATION CHECK ===")
print(f"Brier score: {brier:.4f}  (< 0.05 = excellent, 0.25 = random)")
print(f"\n{'Predicted range':20s} {'Actual scam rate':>18}")
print("-" * 40)
for mp, fp in zip(mean_pred, fraction_pos):
    gap   = fp - mp
    flag  = "⚠️ " if abs(gap) > 0.15 else "✅"
    print(f"  ~{mp:.2f}               {fp:.2f}   {flag}")

if brier < 0.05:
    print("\nVerdict: Well calibrated ✅ — scores are trustworthy")
elif brier < 0.10:
    print("\nVerdict: Acceptable ⚠️ — minor miscalibration, thresholds are still valid")
else:
    print("\nVerdict: Miscalibrated 🚨 — consider isotonic regression calibration")

# ── Tier distribution + precision per tier ─────────────────────────────────────
# Using test_proba directly (already computed) — no new prediction needed
_, _, tiers = predictor.predict_with_tier(X_test)  # tiers only, proba already stored above

print("\n=== TIER BREAKDOWN ===")
tier_series = pd.Series(tiers)
tier_cuts   = pd.cut(
    test_proba,
    bins=[0, 0.50, predictor.threshold, 1.0],
    labels=["CLEAR", "HUMAN_REVIEW", "AUTO_REMOVE"],
    include_lowest=True
)
print(f"{'Tier':15s} {'Total':>7} {'Scam':>7} {'Legal':>7} {'Scam rate':>10}")
for tier in ["AUTO_REMOVE", "HUMAN_REVIEW", "CLEAR"]:
    mask    = tier_cuts == tier
    n       = mask.sum()
    n_scam  = ((tier_cuts == tier) & (y_test == 1)).sum()
    n_legal = ((tier_cuts == tier) & (y_test == 0)).sum()
    rate    = n_scam / n if n > 0 else 0
    print(f"{tier:15s} {n:>7} {n_scam:>7} {n_legal:>7} {rate:>9.1%}")

# ── False negatives ────────────────────────────────────────────────────────────
fn_mask = (y_test == 1) & (test_pred == 0)
fn_df   = test_df[fn_mask].copy()
fn_df["scam_prob"] = test_proba[fn_mask]     # reuse test_proba, no new call
fn_df   = fn_df.sort_values("scam_prob", ascending=True)

print(f"\n=== FALSE NEGATIVES: {fn_mask.sum()} missed scams ===")
for _, row in fn_df.iterrows():
    print(f"\nScam probability : {row['scam_prob']:.4f}")
    print(f"Text             : {row['text']}")
    print("-" * 60)

# ── False positives ────────────────────────────────────────────────────────────
fp_mask = (y_test == 0) & (test_pred == 1)
fp_df   = test_df[fp_mask].copy()
fp_df["scam_prob"] = test_proba[fp_mask]     # reuse test_proba
fp_df   = fp_df.sort_values("scam_prob", ascending=False)

print(f"\n=== FALSE POSITIVES: {fp_mask.sum()} legal listings wrongly flagged ===")
for _, row in fp_df.iterrows():
    print(f"\nScam probability : {row['scam_prob']:.4f}")
    print(f"Text             : {row['text']}")
    print("-" * 60)

# ── LLM explanations for borderline false negatives ───────────────────────────
explainer     = Explainer()
borderline_fn = fn_df[fn_df["scam_prob"] >= 0.30]

print(f"\n=== EXPLANATIONS ({len(borderline_fn)} borderline FN) ===")
for _, row in borderline_fn.iterrows():
    tier        = predictor._tier(row["scam_prob"])
    explanation = explainer.explain(row["text"], row["scam_prob"], tier)
    print(f"\nProb : {row['scam_prob']:.4f}  |  Tier: {tier}")
    print(f"Text : {row['text'][:80]}...")
    print(f"Why  : {explanation}")
    print("-" * 50)

# ── Calibration plot ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 5), facecolor="#1a1a2e")
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# Left: reliability diagram
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor("#16213e")
ax1.plot([0, 1], [0, 1], "--", color="#aaaaaa", linewidth=1.2, label="Perfect calibration")
ax1.plot(mean_pred, fraction_pos, "o-", color="#e94560",
         linewidth=2, markersize=8, label="Our model")
ax1.fill_between(mean_pred, mean_pred, fraction_pos, alpha=0.15, color="#e94560")
ax1.set_xlabel("Mean predicted probability", color="#cccccc", fontsize=11)
ax1.set_ylabel("Fraction of actual scams",   color="#cccccc", fontsize=11)
ax1.set_title("Reliability Diagram", color="white", fontsize=13, fontweight="bold")
ax1.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=9)
ax1.tick_params(colors="#aaaaaa")
for sp in ax1.spines.values(): sp.set_edgecolor("#444444")
ax1.text(0.05, 0.92, f"Brier: {brier:.4f}", transform=ax1.transAxes,
         color="#aaaaaa", fontsize=9)

# Right: score distribution
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor("#16213e")
ax2.hist(test_proba[y_test == 0], bins=30, alpha=0.6,
         color="#4fc3f7", label=f"Legal  (n={( y_test==0).sum()})")
ax2.hist(test_proba[y_test == 1], bins=30, alpha=0.7,
         color="#e94560", label=f"Scam   (n={(y_test==1).sum()})")
ax2.axvline(predictor.threshold, color="white", linestyle="--",
            linewidth=1.5, label=f"Threshold {predictor.threshold:.3f}")
ax2.set_xlabel("Predicted scam probability", color="#cccccc", fontsize=11)
ax2.set_ylabel("Count",                      color="#cccccc", fontsize=11)
ax2.set_title("Score Distribution", color="white", fontsize=13, fontweight="bold")
ax2.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=9)
ax2.tick_params(colors="#aaaaaa")
for sp in ax2.spines.values(): sp.set_edgecolor("#444444")

plt.suptitle("Salla Scam Detector — Calibration & Score Distribution",
             color="white", fontsize=13, fontweight="bold", y=1.02)
plt.savefig("calibration_plot.png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
print("\nCalibration plot saved → calibration_plot.png")