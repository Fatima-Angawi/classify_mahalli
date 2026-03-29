import re
import pandas as pd
from bs4 import BeautifulSoup
import pyarabic.araby as araby


def clean_html(text: str) -> str:
    if pd.isna(text) or text == "":
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=' ').strip()

def preprocess(df: pd.DataFrame) -> pd.DataFrame:

    df['description'] = df['description'].apply(clean_html)
    df['name']        = df['name'].apply(clean_html)

    df['text'] = df['name'] + " " + df['description']



