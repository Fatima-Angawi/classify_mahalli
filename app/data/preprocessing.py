import re
import pandas as pd
from bs4 import BeautifulSoup
import pyarabic.araby as araby


def clean_html(text: str) -> str:
    if pd.isna(text) or text == "":
        return ""
    text =BeautifulSoup(text, "html.parser").get_text(separator=' ').strip()
    text = araby.strip_tashkeel(text)

    text = re.sub(r"[^\w\sء-ي]", " ", text)  
    text = re.sub(r"\s+", " ", text)
    return text

def combine_texts(df: pd.DataFrame) -> pd.DataFrame:

    df['description'] = df['description'].apply(clean_html)
    df['name']        = df['name'].apply(clean_html)

    df['text'] = df['name'] + " " + df['description']

    df = df[['text']].copy() #add price,thumbnail,updated_date
    
    return df
def validate_preview(df):
    print(df.head(5))
    proceed = input("Does the preview look correct? (y/n): ").strip().lower()
    return proceed == 'y'
        





