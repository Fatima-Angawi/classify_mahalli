import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    return df