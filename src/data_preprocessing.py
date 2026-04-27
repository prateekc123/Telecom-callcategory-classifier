import pandas as pd
import re

def preprocess_data(file_path):
    return pd.read_csv(file_path)

def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-zA-Z0-9\s]','',text)
    return text

def preprocess(df):
    df=df.dropna(subset=['text','label'])
    df=df[df['text'].str.strip()!='']
    df=df.drop_duplicates()
    
    df['label']=df['label'].str.strip().str.title()
    df['clean_text']=df['text'].apply(clean_text)

    return df

