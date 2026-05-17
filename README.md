# Arabic Hotel Reviews — Sentiment Analysis

Sentiment analysis on Arabic hotel reviews comparing a classical Machine Learning model (Logistic Regression + TF-IDF) with a Deep Learning model (Bidirectional LSTM) on the **HARD dataset** (105,645 reviews from Booking.com).

> NLP Course Project — Natural Language Processing
> Comparing classical ML and Deep Learning approaches on Arabic text.

---

## 📊 Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| **Logistic Regression + TF-IDF** | **84.5%** | **0.845** |
| Bidirectional LSTM | 83.0% | 0.829 |

> **Key finding:** The simpler classical model outperformed the deep learning model by ~1.5 points.

---

## 📁 Dataset

- **Name:** HARD (Hotel Arabic Reviews Dataset)
- **Source:** [Kaggle](https://www.kaggle.com/datasets/mksaad/arabic-hotel-reviews-dataset)
- **Size:** 105,645 reviews
- **Language:** Modern Standard Arabic + dialects
- **Classes:** Positive / Negative (balanced ~50/50)

### Label Mapping
| Rating | Label |
|--------|-------|
| 1–2 | Negative (0) |
| 3 | Excluded |
| 4–5 | Positive (1) |

---

## 🧹 Preprocessing

1. Remove Latin characters and digits
2. Remove Arabic diacritics (تشكيل)
3. Normalize alef variants (إ، أ، آ → ا)
4. Normalize letters (ة → ه، ى → ي)
5. Remove punctuation and tatweel
6. Remove Arabic stopwords

---

## 🤖 Models

### Model 1: Logistic Regression + TF-IDF
- `max_features`: 20,000
- `ngram_range`: (1, 2)
- `min_df`: 2
- `max_iter`: 1000, `C`: 1.0

### Model 2: Bidirectional LSTM
- Embedding (64 dim) → BiLSTM (64 units) → Dropout (0.3) → Dense (32, ReLU) → Dropout (0.3) → Dense (1, Sigmoid)

---

## ⚙️ Experimental Setup

- **Train/Test Split:** 80% / 20% (stratified)
- **Training set:** 84,516 reviews
- **Test set:** 21,129 reviews
- **Tools:** Python 3, Google Colab, scikit-learn, TensorFlow/Keras

---

## 💡 Why Logistic Regression Won

1. Reviews are short (5–30 tokens) — limits LSTM's advantage
2. Strong polarized words (ممتاز، سيء) dominate the signal
3. TF-IDF with bigrams captures key phrases
4. Embedding trained from scratch — limited capacity

---

## 🚀 Future Work

- Try **AraBERT** or **MARBERT** — pretrained Arabic transformers
- Experiment with dialect-specific preprocessing

---

## 📦 Requirements

```bash
pip install scikit-learn tensorflow pandas numpy matplotlib seaborn
