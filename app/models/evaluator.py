import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve

def find_best_threshold_tc(model, texts_val, y_val, min_recall=0.8):
    y_proba = model.predict(texts_val)  # probabilities from TextClassifier


    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
    mask = recalls[:-1] >= min_recall
    if not mask.any():
        raise ValueError(f"No threshold found with recall >= {min_recall}")

    best_idx = precisions[:-1][mask].argmax()
    return thresholds[mask][best_idx]

def evaluate_tc(model, texts, y_true, threshold, name="Test"):
    # Predict probabilities
    y_proba = model.predict(texts)  # ensure this returns probabilities
    # Convert probabilities to binary predictions using the threshold
    y_pred = (y_proba >= threshold).astype(int)
    
    # Compute AUC-ROC
    auc = roc_auc_score(y_true, y_proba)
    
    # Print header
    print(f"\n{'='*45}")
    print(f"{name} Set — AUC-ROC: {auc:.4f} | Threshold: {threshold:.4f}")
    
    # Print classification report (precision, recall, F1)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Legal', 'Scam']))
    
    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

"""
def find_best_threshold(model, X_val, y_val, min_recall=0.8):
    y_proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)

    mask = recalls[:-1] >= min_recall
    if not mask.any():
        raise ValueError(f"No threshold found with recall >= {min_recall}")

    best_idx = precisions[:-1][mask].argmax()
    return thresholds[mask][best_idx]


def evaluate(model, X_val, y_val, X_test, y_test, threshold):
    for name, X, y in [("Validation", X_val, y_val), ("Test", X_test, y_test)]:
        y_proba = model.predict_proba(X)[:, 1]
        y_pred  = (y_proba >= threshold).astype(int)
        auc     = roc_auc_score(y, y_proba)   # fix 3: AUC-ROC

        print(f"\n{'='*45}")
        print(f"{name} Set — AUC-ROC: {auc:.4f} | Threshold: {threshold:.4f}")
        print(classification_report(y, y_pred, target_names=['Legal', 'Scam']))


def evaluate_with_realistic_distribution(model, X_test, y_test, threshold, scam_ratio=0.1):
    scam_idx  = np.where(y_test == 1)[0]
    legit_idx = np.where(y_test == 0)[0]

    n_scam = int(len(legit_idx) * scam_ratio / (1 - scam_ratio))
    n_scam = min(n_scam, len(scam_idx))

    rng          = np.random.default_rng(42)
    scam_sampled = rng.choice(scam_idx, size=n_scam, replace=False)
    idx          = np.concatenate([legit_idx, scam_sampled])

    X_real, y_real = X_test[idx], y_test[idx]
    y_proba = model.predict_proba(X_real)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    print(f"\n{'='*45}")
    print(f"Realistic Distribution — legit: {len(legit_idx)}, scam: {n_scam} ({scam_ratio*100:.0f}%)")
    print(f"AUC-ROC: {roc_auc_score(y_real, y_proba):.4f}")
    print(classification_report(y_real, y_pred, target_names=['Legal', 'Scam']))
    """