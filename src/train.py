# src/train.py
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch, pandas as pd
from sklearn.model_selection import train_test_split

# --- Dataset class ---
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# --- Load data ---
df = pd.read_csv('data/processed_tweets.csv').sample(50000)  # start small
X_train, X_val, y_train, y_val = train_test_split(
    df['clean_text'].tolist(), df['label'].tolist(), test_size=0.2)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

train_ds = TweetDataset(X_train, y_train, tokenizer)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

# --- Training loop ---
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=batch['label'].to(device)
        )
        outputs.loss.backward()
        optimizer.step()
        total_loss += outputs.loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

model.save_pretrained('models/bert_sentiment')
tokenizer.save_pretrained('models/bert_sentiment')