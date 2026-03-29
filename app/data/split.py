import pandas as pd
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE, TEST_SIZE, VAL_SIZE


def temporal_split(df, test_size, val_size):
    """
    Temporal split to simulate real-world deployment:
    train on past → validate on more recent → test on latest data
    """
    df = df.sort_values("updated_date")

    n = len(df)

    # define cutoff for test (latest portion of the data)
    test_cut = int(n * (1 - test_size))

    # validation is computed relative to the full dataset
    # not just the remaining training portion
    val_cut = int(test_cut * (1 - val_size / (1 - test_size)))

    train = df.iloc[:val_cut]
    val   = df.iloc[val_cut:test_cut]
    test  = df.iloc[test_cut:]

    return train, val, test


def split_df(df: pd.DataFrame):
    """
    Splitting strategy depends on data availability:

    - If timestamps are widely available:
        → use full temporal split to avoid leakage and reflect production

    - If timestamps are missing (especially for scam class):
        → use hybrid approach:
            legal → temporal (time-aware)
            scam  → stratified random (preserve class distribution)
    """

    # evaluate how usable the time feature is
    dated_ratio = df["updated_date"].notna().mean()
    print(f"Dated ratio: {dated_ratio:.0%}")

    # =========================================================
    # Case 1: timestamps are reliable across the dataset
    # =========================================================
    if dated_ratio > 0.9:
        print("Using FULL TEMPORAL split")

        # when both classes have timestamps,
        # splitting by class is unnecessary and less realistic
        df_dated = df[df["updated_date"].notna()].copy()

        train_df, val_df, test_df = temporal_split(
            df_dated, TEST_SIZE, VAL_SIZE
        )

    # =========================================================
    # Case 2: timestamps are incomplete (current scenario)
    # =========================================================
    else:
        print("Using HYBRID split")

        # separate classes due to different data constraints
        legal = df[df["label"] == 0].copy()
        scam  = df[df["label"] == 1].copy()

        # ── Legal: time-aware split ─────────────────
        # timestamps are mostly available → use temporal split
        legal_dated   = legal[legal["updated_date"].notna()]
        legal_undated = legal[legal["updated_date"].isna()]

        legal_train_d, legal_val, legal_test = temporal_split(
            legal_dated, TEST_SIZE, VAL_SIZE
        )

        # undated samples are added to training only
        # to avoid contaminating validation/test temporal ordering
        legal_train = pd.concat([legal_train_d, legal_undated])

        # ── Scam: distribution-aware split ──────────
        # timestamps are mostly missing → temporal split not reliable
        # instead, enforce consistent class distribution via stratification

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

        # ── Combine both strategies ─────────────────
        train_df = pd.concat([legal_train, scam_train])
        val_df   = pd.concat([legal_val,   scam_val])
        test_df  = pd.concat([legal_test,  scam_test])

    # final shuffle to remove ordering bias before training
    train_df = train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    #dont shuffle val/test to preserve temporal structure for evaluation
    
    # sanity check: monitor class distribution across splits
    print("\nSplit distribution:")
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        legal_count = (split["label"] == 0).sum()
        scam_count  = (split["label"] == 1).sum()
        print(f"{name} — Legal: {legal_count} | Scam: {scam_count}")

    return train_df, val_df, test_df