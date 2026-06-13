# src/collect.py
import pandas as pd

# Load Sentiment140 dataset (download from Kaggle)
df = pd.read_csv('data/training.1600000.processed.noemoticon.csv',
                 encoding='latin-1', header=None)
df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# Map labels: 0 = negative, 4 = positive → convert to 0/1
df['label'] = df['sentiment'].map({0: 0, 4: 1})
df = df[['text', 'label']]
df.to_csv('data/clean_tweets.csv', index=False)