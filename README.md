# 🐦 Social Media Sentiment Analyzer

A real-time sentiment analysis dashboard built with Streamlit and BERT, designed to analyze the emotional tone of social media posts and text snippets.

## Features

- ✨ **Real-time Analysis** — Instantly classify text as positive or negative sentiment
- 🚀 **BERT-Powered** — Uses DistilBERT pre-trained model for accurate sentiment detection
- 🎯 **User-Friendly UI** — Simple, intuitive Streamlit interface with emoji indicators
- ⚡ **Fast Processing** — Optimized model caching for instant predictions
- 📊 **Production-Ready** — Includes data preprocessing, training, and evaluation scripts

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone or download the repository**
   ```bash
   cd sentiment-dashboard
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows
   # or: source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### How to Use

1. Enter a tweet or social media post in the text area
2. Click the **"Analyze"** button
3. View the sentiment result (😊 Positive or 😡 Negative)

### Example Inputs

- **Positive:** "I love this product! It's amazing and works perfectly!"
- **Negative:** "This is terrible and doesn't work at all."
- **Neutral:** "The weather is cloudy today."

## Project Structure

```
sentiment-dashboard/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── sample_data.csv       # Example tweets for batch testing
├── screenshots/          # Demo screenshots
├── src/
│   ├── collect.py        # Data collection from Sentiment140 dataset
│   ├── preprocess.py     # Text preprocessing and cleaning
│   ├── train.py          # Model training script
│   └── predict.py        # Prediction and evaluation utilities
├── data/
│   └── clean_tweets.csv  # Processed training data (if available)
└── models/
    └── bert_sentiment/   # Trained model directory
```

## Dependencies

- **streamlit** — Web app framework
- **transformers** — Hugging Face models
- **torch** — PyTorch deep learning
- **pandas** — Data manipulation
- **scikit-learn** — ML utilities (for training/evaluation)
- **seaborn** — Visualization
- **matplotlib** — Plotting

See `requirements.txt` for exact versions.

## Model Details

- **Base Model:** DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
- **Task:** Binary Sentiment Classification
- **Classes:** Negative (0) | Positive (1)
- **Max Input Length:** 128 tokens
- **Framework:** PyTorch + Hugging Face Transformers

## Dataset

The project uses the **Sentiment140 dataset** (available on Kaggle):
- 1.6 million labeled tweets
- Labels: 0 (negative) and 4 (positive)
- Preprocessed and mapped to 0/1 labels

To use your own data:
1. Place CSV files in `data/` directory
2. Update `src/collect.py` to parse your format
3. Run preprocessing: `python src/preprocess.py`

## Performance

- **Accuracy:** ~95% on SST-2 benchmark
- **Inference Speed:** ~50-100ms per tweet
- **Model Size:** ~67MB (DistilBERT)

## Advanced Features

### Batch Processing
You can extend `app.py` to process multiple texts from `sample_data.csv`:
```python
# Add to app.py
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    predictions = df['text'].apply(lambda x: predict(x, tokenizer, model))
```

### Confidence Scores
Modify `predict()` to return probability scores:
```python
probs = torch.softmax(outputs.logits, dim=1)
confidence = probs[0][prediction].item()
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Module not found | Run `pip install -r requirements.txt` |
| Slow first load | Model caching activates after first use |
| Out of memory | Reduce `max_length` in `predict()` function |
| CUDA errors | Falls back to CPU automatically |

## Contributing

Feel free to fork, improve, and submit pull requests! Ideas:
- Multi-class sentiment (neutral support)
- Aspect-based sentiment analysis
- Batch CSV upload
- Docker containerization
- API endpoint wrapper

## License

MIT License — Free for personal and commercial use.

## References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Sentiment140 Dataset](http://www.sentiment140.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## Contact & Support

For questions or issues, please open an issue in the repository.

---

**Built with ❤️ using BERT and Streamlit**
