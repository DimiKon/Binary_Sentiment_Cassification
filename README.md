
# Sentiment Analysis using Deep Learning

This repository contains implementations for sentiment analysis using various models and preprocessing techniques. The dataset used is the **Large Movie Review Dataset (IMDB)**, and the models explored include **CNN**, **LSTM**, and **DistilBERT**.

---

## Project Structure

- **`CNN_Sentiment.ipynb`**: Implementation of a Convolutional Neural Network (CNN) for sentiment classification.
- **`LSTM_Sentiment.ipynb`**: Implementation of a Long Short-Term Memory (LSTM) network for sentiment analysis.
- **`DistilBERT_Sentiment.ipynb`**: Fine-tuning a DistilBERT model for sentiment classification.
- **`Large_Movie_Review_Dataset_preprocessing_n_splitting.py`**: Script for preprocessing the IMDB dataset and splitting it into training, validation, and test sets.

---

## Dataset

The **Large Movie Review Dataset (IMDB)** is used for training and evaluation:
- Positive and negative movie reviews are preprocessed and cleaned.
- The dataset is split into:
  - Training set: 82%
  - Validation set: 9%
  - Test set: 9%

Preprocessing steps:
1. Removing HTML tags.
2. Expanding contractions (e.g., "can't" → "cannot").
3. Removing punctuation and numerals.
4. Tokenizing and cleaning text.

---

## Models

1. **CNN for Sentiment Analysis**
   - Captures local features using convolutional layers.
   - Implemented in `CNN_Sentiment.ipynb`.

2. **LSTM for Sentiment Analysis**
   - Explores sequential dependencies in text.
   - Implemented in `LSTM_Sentiment.ipynb`.

3. **DistilBERT Fine-Tuning**
   - Uses the transformer-based DistilBERT model for sentiment classification.
   - Implemented in `DistilBERT_Sentiment.ipynb`.

---

## How to Run

1. **Preprocess Dataset**
   - Run `Large_Movie_Review_Dataset_preprocessing_n_splitting.py` to generate cleaned and split CSV files.
   - Outputs:
     - `data_train.csv`
     - `data_dev.csv`
     - `data_test.csv`

2. **Train Models**
   - Use the respective Jupyter notebooks (`CNN_Sentiment.ipynb`, `LSTM_Sentiment.ipynb`, `DistilBERT_Sentiment.ipynb`) for model training and evaluation.

3. **Evaluate**
   - Each notebook includes evaluation metrics and loss/accuracy plots for performance analysis.

---

## Dependencies

Install the required libraries:
```bash
pip install pandas seaborn scikit-learn torch transformers
```

---

## Results

Each model achieves sentiment classification on the IMDB dataset, with varying levels of accuracy:
- CNN and LSTM capture sequential patterns effectively.
- DistilBERT demonstrates state-of-the-art performance.

---


## Author

Created by **dim_k**.
