REQUIRED_COLUMNS = ["text", "label"]
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
import pandas as pd
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ── Validate required columns ──────────────────────────────
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing required column: '{col}'"

    assert len(df) > 0, "Dataset is empty"

    # ── Handle nulls ───────────────────────────────────────────
    for col in ["text"]:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"Warning: {missing} null values in '{col}' — filling with empty string")
        df[col] = df[col].fillna("")

        

    # ── Parse date if exists ───────────────────────────────────
    if "updated_date" in df.columns:
        df["updated_date"] = pd.to_datetime(df["updated_date"], errors="coerce")


    # ── Remove conflicting labels & duplicates ─────────────────
    conflict = df.groupby("text")["label"].nunique()
    conflict_texts = conflict[conflict > 1].index
    if len(conflict_texts) > 0:
        print(f"Removing {len(conflict_texts)} texts with conflicting labels")
        df = df[~df["text"].isin(conflict_texts)]

    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        print(f"Removed {removed} duplicate texts")

    print(f"Loaded {len(df)} rows | Labels: {df['label'].value_counts().to_dict()}")

    return df
