# src/preprocess.py
import re
import pandas as pd

def clean_text(text):
    text = re.sub(r'http\S+', '', text)        # remove URLs
    text = re.sub(r'@\w+', '', text)           # remove mentions
    text = re.sub(r'#', '', text)              # remove hashtag symbol
    text = re.sub(r'[^A-Za-z\s]', '', text)   # keep only letters
    return text.lower().strip()

df = pd.read_csv('data/clean_tweets.csv')
df['clean_text'] = df['text'].apply(clean_text)
df.dropna(inplace=True)
df.to_csv('data/processed_tweets.csv', index=False)