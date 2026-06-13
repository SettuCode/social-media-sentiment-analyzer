import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

@st.cache_resource
def load_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    model.eval()
    return tokenizer, model

def predict(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return "😊 Positive" if prediction == 1 else "😡 Negative"

# UI
st.title("🐦 Social Media Sentiment Analyzer")
st.subheader("Powered by BERT")

tokenizer, model = load_model()

user_input = st.text_area(
    "Paste a tweet or social media post:",
    height=120
)

if st.button("Analyze"):
    if user_input.strip():
        result = predict(user_input, tokenizer, model)
        st.success(f"Sentiment: {result}")
    else:
        st.warning("Please enter some text.")