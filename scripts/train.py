import os
import sys
from pathlib import Path
import gc
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from app.data.loader import load_dataset
from app.models.Classifier import TextClassifier
from app.data.split import split_df
os.environ["PYTHONUTF8"] = "1"
from sklearn.utils.class_weight import compute_class_weight

from app.models.evaluator import evaluate_tc, find_best_threshold_tc
# ── Load ──────────────────────────────────────────────────
df = load_dataset("data/mahalli_combined_text.csv")

train_df, val_df, test_df = split_df(df)

# ── preprocess ──────────────────────────────────────────────────

Text_Classifier = TextClassifier()
#to avoid data leakage, we only preprocess text after splitting
for split in [train_df, val_df, test_df]:
    split["text"] = split["text"].astype(str).apply(
        lambda x: "".join(c for c in x if c.isprintable())
    )
    split["text"] = split["text"].apply(Text_Classifier.preprocess)

neg, pos = np.bincount(train_df["label"].values)
weights  = compute_class_weight("balanced", classes=np.array([0,1]), y=train_df["label"].values)
class_weights_tensor = torch.tensor(weights, dtype=torch.float)

# Fine-tune
Text_Classifier.fine_tune(train_df, val_df, class_weights_tensor)

gc.collect()
torch.cuda.empty_cache()
# evaluate
threshold = find_best_threshold_tc(Text_Classifier, val_df["text"].tolist(), val_df["label"].values)
evaluate_tc(Text_Classifier, test_df["text"].tolist(), test_df["label"].values, threshold=threshold, name="Test")

#Text_Classifier.set_threshold(threshold)
#preds, probs, tiers = Text_Classifier.predict_with_tier(test_df["text"].tolist())
# save model
Text_Classifier.save("artifacts/AraBERT")


"""
# ── Train ─────────────────────────────────────────────────
trainer = Trainer()
trainer.train(X_train, y_train, X_val, y_val)

# ── Evaluate ──────────────────────────────────────────────
threshold = find_best_threshold(trainer.model, X_val, y_val, min_recall=0.8)
print(f"Selected threshold: {threshold:.4f}")

evaluate(trainer.model, X_val, y_val, X_test, y_test, threshold)
evaluate_with_realistic_distribution(trainer.model, X_test, y_test, threshold, scam_ratio=0.1)

# ── Save ──────────────────────────────────────────────────
trainer.save(threshold)
"""
