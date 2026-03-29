"""
Plots a reliability diagram for the XGBoost scam detector.
Run from project root: python calibration_check.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from app.data.loader import load_dataset
from app.data.split import split_df
from app.embeddings.embedder import Embedder
from app.inference.predictor import Predictor


# ── Load model & data ─────────────────────────────────────────────────────────
predictor = Predictor()
print(f"Loaded model | threshold: {predictor.threshold:.4f}")

df = load_dataset("data/combined_data_final.csv")
_, test_df = split_df(df)

embedder = Embedder()
X_test = np.array(embedder.encode(test_df["text"].astype(str).tolist()))
y_test = test_df["label"].values

# ── Raw scores ────────────────────────────────────────────────────────────────
y_proba = predictor.model.predict_proba(X_test)[:, 1]

# ── Calibration curve ─────────────────────────────────────────────────────────
# n_bins=8 because we only have ~137 scams — more bins = too few samples per bucket
fraction_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=8, strategy="uniform")

# Brier score: lower = better (0 = perfect, 0.25 = random for balanced classes)
brier = brier_score_loss(y_test, y_proba)
print(f"Brier score: {brier:.4f}  (lower is better, <0.05 is excellent)")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 5), facecolor="#1a1a2e")
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# --- Left: Reliability diagram ---
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor("#16213e")

ax1.plot([0, 1], [0, 1], linestyle="--", color="#aaaaaa", linewidth=1.2, label="Perfect calibration")
ax1.plot(mean_pred, fraction_pos, marker="o", color="#e94560",
         linewidth=2, markersize=8, label="XGBoost (our model)")

# Shade the gap between model and perfect
ax1.fill_between(mean_pred, mean_pred, fraction_pos,
                 alpha=0.15, color="#e94560", label="Calibration gap")

ax1.set_xlabel("Mean predicted probability", color="#cccccc", fontsize=11)
ax1.set_ylabel("Fraction of actual scams", color="#cccccc", fontsize=11)
ax1.set_title("Reliability Diagram", color="white", fontsize=13, fontweight="bold", pad=12)
ax1.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=9)
ax1.tick_params(colors="#aaaaaa")
for spine in ax1.spines.values():
    spine.set_edgecolor("#444444")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

# Annotate Brier score
ax1.text(0.05, 0.92, f"Brier score: {brier:.4f}", transform=ax1.transAxes,
         color="#aaaaaa", fontsize=9)

# --- Right: Score distribution ---
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor("#16213e")

scam_probs  = y_proba[y_test == 1]
legal_probs = y_proba[y_test == 0]

ax2.hist(legal_probs, bins=30, alpha=0.6, color="#4fc3f7", label=f"Legal  (n={len(legal_probs)})")
ax2.hist(scam_probs,  bins=30, alpha=0.7, color="#e94560", label=f"Scam   (n={len(scam_probs)})")

# Threshold line
ax2.axvline(predictor.threshold, color="white", linestyle="--",
            linewidth=1.5, label=f"Threshold {predictor.threshold:.3f}")

ax2.set_xlabel("Predicted scam probability", color="#cccccc", fontsize=11)
ax2.set_ylabel("Count", color="#cccccc", fontsize=11)
ax2.set_title("Score Distribution", color="white", fontsize=13, fontweight="bold", pad=12)
ax2.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=9)
ax2.tick_params(colors="#aaaaaa")
for spine in ax2.spines.values():
    spine.set_edgecolor("#444444")

plt.suptitle("Salla Scam Detector — Calibration Check", color="white",
             fontsize=14, fontweight="bold", y=1.02)

out_path = "calibration_plot.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
print(f"\nSaved → {out_path}")

# ── Text summary ──────────────────────────────────────────────────────────────
print("\n=== SCORE DISTRIBUTION SUMMARY ===")
print(f"{'':20s} {'Min':>8} {'Median':>8} {'Max':>8}")
print(f"{'Legal scores':20s} {legal_probs.min():>8.4f} {np.median(legal_probs):>8.4f} {legal_probs.max():>8.4f}")
print(f"{'Scam scores':20s} {scam_probs.min():>8.4f}  {np.median(scam_probs):>8.4f} {scam_probs.max():>8.4f}")

# How many fall in each tier
tiers = pd.cut(y_proba,
               bins=[0, 0.50, predictor.threshold, 1.0],
               labels=["CLEAR", "HUMAN_REVIEW", "AUTO_REMOVE"],
               include_lowest=True)

print("\n=== TIER BREAKDOWN (test set) ===")
for tier in ["AUTO_REMOVE", "HUMAN_REVIEW", "CLEAR"]:
    mask       = tiers == tier
    n_total    = mask.sum()
    n_scam     = ((tiers == tier) & (y_test == 1)).sum()
    n_legal    = ((tiers == tier) & (y_test == 0)).sum()
    precision  = n_scam / n_total if n_total > 0 else 0
    print(f"{tier:15s}  total={n_total:4d}  scam={n_scam:3d}  legal={n_legal:4d}  scam_rate={precision:.1%}")