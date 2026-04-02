import pandas as pd
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE, TEST_SIZE, VAL_SIZE

def split_df(df: pd.DataFrame):
    """
    Split dataset into train/val/test without using timestamps.
    Stratified split is used to preserve class distribution.
    """

    # Separate features by label
    legal = df[df["label"] == 0].copy()
    scam  = df[df["label"] == 1].copy()

    # ── Split Legal ──
    legal_train_val, legal_test = train_test_split(
        legal,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=legal["label"]
    )
    legal_train, legal_val = train_test_split(
        legal_train_val,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE,
        stratify=legal_train_val["label"]
    )

    # ── Split Scam ──
    scam_train_val, scam_test = train_test_split(
        scam,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=scam["label"]
    )
    scam_train, scam_val = train_test_split(
        scam_train_val,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE,
        stratify=scam_train_val["label"]
    )

    # ── Combine splits ──
    train_df = pd.concat([legal_train, scam_train]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    val_df   = pd.concat([legal_val, scam_val]).reset_index(drop=True)
    test_df  = pd.concat([legal_test, scam_test]).reset_index(drop=True)

    # Print class distribution
    print("\nSplit distribution (No date):")
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        legal_count = (split["label"] == 0).sum()
        scam_count  = (split["label"] == 1).sum()
        print(f"{name} — Legal: {legal_count} | Scam: {scam_count}")

    return train_df, val_df, test_df
