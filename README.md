# Amazon Product Reviews Sentiment Analysis

Does a fine-tuned transformer really beat a well-tuned classical model on review sentiment, and by how much? This project answers that on 25,000 Amazon product reviews, comparing TF-IDF + Logistic Regression against a fine-tuned DistilBERT on the same train/test split.

## Why I built it

I wanted a fair comparison, not just "BERT is better". It is easy to put a transformer next to a lazily built baseline and call it a win, so I tuned the classical model properly first (cross-validated model selection, joint grid search, a calibration check) and only then fine-tuned DistilBERT on exactly the same split. I also kept the neutral 3-star reviews in a separate 3-class run to see what ambiguity actually costs.

## Results

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| TF-IDF + Logistic Regression (tuned) | 0.87 | 0.874 | 0.943 |
| DistilBERT (fine-tuned, 2 epochs) | 0.92 | 0.919 | - |

- Best classical settings: `C=1`, `max_features=8000`, `ngram_range=(1,2)`.
- The classical model's probabilities are reasonably calibrated (Brier score about 0.10). I did not check calibration for DistilBERT.
- 3-class run (negative / neutral / positive): macro F1 about 0.63. Almost all of the loss comes from the neutral class (F1 about 0.40), which is genuinely hard to separate from both sides.

## What I did

1. **Cleaned the data:** dropped nulls and 255 duplicate reviews, removed 3-star reviews for the binary task, and mapped 4-5 stars to positive and 1-2 stars to negative. That left 19,784 reviews at roughly 50/50.
2. **Preprocessed the text:** lowercased, lemmatized and removed stopwords, but kept negations like "not" and "never", because dropping them flips the meaning of "not good".
3. **Avoided leakage:** the TF-IDF vectorizer sits inside a scikit-learn `Pipeline`, so it is only ever fit on training data, including inside cross-validation and grid search.
4. **Chose a model:** 5-fold stratified CV across Logistic Regression, Multinomial Naive Bayes and Linear SVC, then a joint grid search over vectorizer and model settings.
5. **Evaluated on a held-out set:** classification report, ROC-AUC, confusion matrix, most predictive terms, and a calibration curve.
6. **Fine-tuned DistilBERT** on the same split and compared it directly with the classical model.

![Classical vs transformer](07_classical_vs_transformer.png)

## Files

- `amazon-sentiment-analysis.ipynb`: the full pipeline (EDA, cleaning, classical models, DistilBERT, evaluation).
- `01_...png` to `07_...png`: the plots from the notebook.

## Running it

The review CSV is not included in this repo. Put it next to the notebook as `Amazon-Product-Reviews-Sentiment-Analysis-in-Python-Dataset.csv` (columns: `Review`, `Sentiment` as a 1-5 star rating, 5,000 reviews per rating).

Classical part: `pandas numpy matplotlib seaborn wordcloud nltk scikit-learn`.
DistilBERT part (needs a GPU, for example Colab or Kaggle): `torch transformers datasets accelerate evaluate`.

## Limitations

- DistilBERT sees raw review text while the classical pipeline sees cleaned, lemmatized text, because transformer tokenizers handle subwords themselves.
- 3-star reviews are left out of the binary task on purpose; the 3-class run covers them.
- DistilBERT's confidence scores were not calibrated and are probably overconfident.
