# Yelp Sentiment Classifier

This project performs sentiment analysis on Yelp reviews using a pre-trained transformer model from Hugging Face. It processes and visualizes a dataset of labeled reviews, evaluates model performance, and plots a confusion matrix.

---

## Project Structure

## Features

- Loads and explores a Yelp reviews dataset.
- Visualizes distribution of positive vs negative reviews.
- Uses Hugging Face's `distilbert-base-uncased-finetuned-sst-2-english` for sentiment classification.
- Evaluates accuracy using `sklearn` and displays a confusion matrix.

---

## Requirements

Install the required libraries with:

```bash
pip install pandas matplotlib seaborn scikit-learn transformers
