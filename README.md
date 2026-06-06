# Fine-Grained Sentiment Analysis on Amazon Product Reviews

> Binary sentiment classification of Amazon product review titles using an LSTM neural network, built and run in Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mageswaran4363/FINE-GRAINED-SENTIMENT-ANALYSIS-ON-AMAZON-PRODUCT-REVIEWS/blob/main/sentiment_analysis.ipynb)

---

## Overview

This project classifies Amazon product review titles as **positive (1)** or **negative (0)** using a deep learning pipeline built with TensorFlow/Keras. The model uses an Embedding layer followed by an LSTM to capture sequential patterns in short review text.

The dataset consists of 199 labelled review titles for Bluetooth earphones/headphones scraped from Amazon.

---

## Dataset

| Property | Detail |
|----------|--------|
| Source | Amazon product reviews (Bluetooth earphones/headphones) |
| Format | CSV (`test.csv`) |
| Size | 199 samples |
| Columns | `ReviewTitle` (text), `class` (0 = negative, 1 = positive) |
| Class split | ~50% positive, ~50% negative |

**Example reviews:**

| ReviewTitle | Class |
|-------------|-------|
| "Just awesome wireless headphone under 1000" | 1 (Positive) |
| "Really good and durable" | 1 (Positive) |
| "Worst earphone of my life" | 0 (Negative) |
| "Stopped working in 14 days" | 0 (Negative) |

---

## Pipeline

### 1. Text Preprocessing
- Custom tokenisation with stop word removal (`a`, `the`, `is`, `of`, `and`, `are`, `we`, `s`, `do`, `to`, `but`, `I`, `in`, `they`, `will`, `it`)
- Lowercasing and non-alphanumeric character removal using regex
- **Vocabulary size:** 356 unique words found across the corpus

### 2. Feature Encoding
- Keras `Tokenizer` with `num_words=200`, fitted on the tokenised corpus
- Encoded using **TF-IDF mode** (`texts_to_matrix`)
- Sequences padded to `maxlen=200`

### 3. Model Architecture

```
Embedding(100, 32, input_length=200)
  → SpatialDropout1D(0.25)
  → LSTM(50, dropout=0.5, recurrent_dropout=0.5)
  → Dropout(0.2)
  → Dense(1, activation='sigmoid')
```

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Embedding | (None, 200, 32) | 3,200 |
| SpatialDropout1D | (None, 200, 32) | 0 |
| LSTM | (None, 50) | 16,600 |
| Dropout | (None, 50) | 0 |
| Dense (sigmoid) | (None, 1) | 51 |
| **Total** | | **19,851** |

- **Loss:** Binary Cross-Entropy  
- **Optimiser:** Adam  
- **Epochs:** 10  
- **Batch size:** 32  
- **Validation split:** 20%

### 4. Inference

A `predict_sentiment(text)` function tokenises and pads any new input sentence, then predicts the label using the trained model:

```python
predict_sentiment("good quality and worth buying")
# Predicted label: [1]
```

---

## Results

Training accuracy stabilised at approximately **62%** across all 10 epochs on the training set. The small dataset size (199 samples) and highly imbalanced class representation in the validation split caused the validation accuracy to remain near 0% — indicating the model would benefit from a larger, more balanced dataset.

| Metric | Value |
|--------|-------|
| Training accuracy | ~62% |
| Epochs | 10 |
| Dataset size | 199 |

---

## Running the Notebook

The notebook is designed for **Google Colab**:

1. Click the **Open in Colab** badge above
2. Upload `test.csv` when prompted by the file upload cell
3. Run all cells in order

### Dependencies

All dependencies are pre-installed in Colab. Key packages:

```
tensorflow >= 2.x
keras
pandas
numpy
matplotlib
```

---

## Repository Structure

```
├── sentiment_analysis.ipynb    # Main Colab notebook
└── test.csv                    # Labelled review dataset (upload manually in Colab)
```

---

## Limitations & Future Work

- **Small dataset** — 199 samples is insufficient for reliable generalisation; expanding to a larger Amazon review dataset (e.g. from Kaggle) would significantly improve performance
- **Binary classification** — extending to fine-grained 5-class (1–5 star) sentiment would better justify the project title
- **Pre-trained embeddings** — replacing the learned Embedding layer with GloVe or Word2Vec embeddings would improve representation quality
- **Transformer models** — applying BERT or DistilBERT fine-tuning would likely yield substantially higher accuracy on review text

---

## Author

**Mageswaran Palani**  
MSc Applied Data Science, University of Essex  
[GitHub](https://github.com/Mageswaran4363) | [LinkedIn](https://www.linkedin.com/in/mages-waran/)
