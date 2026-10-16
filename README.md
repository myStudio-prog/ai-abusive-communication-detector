# AI-Powered Abusive Communication Detector

**End-to-End NLP & Machine Learning Project | Trust & Safety AI | Content Moderation**

A deployed machine learning application that analyzes user-generated text, identifies potentially harmful communication across six toxicity categories, and provides human-review recommendations.

**Live Application:** https://ai-abusive-communication-detector-a1.streamlit.app/

**GitHub Repository:** https://github.com/myStudio-prog/ai-abusive-communication-detector

## 1. Project Overview

Online platforms receive large volumes of user-generated communications. Manually reviewing every message is difficult and resource-intensive.

This project demonstrates how Natural Language Processing (NLP) and supervised machine learning can assist content moderation by identifying potentially toxic messages and supporting human-review prioritization.

The application is an experimental decision-support system, not an automated enforcement tool.

## 2. Key Features

- Multi-label classification across six toxicity categories.
- TF-IDF text vectorization and Logistic Regression classification.
- Independent training, validation, and test datasets.
- Category-specific threshold optimization.
- Precision, recall, F1-score, and confusion-matrix evaluation.
- False-positive and false-negative analysis.
- Interactive Streamlit application.
- Human-review recommendations.
- Deployment using Streamlit Community Cloud.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Dataset | Jigsaw Toxic Comment Classification |
| Data processing | Pandas, NumPy |
| NLP feature extraction | TF-IDF |
| Machine learning | Scikit-learn, Logistic Regression |
| Classification | One-vs-Rest multi-label classification |
| Model persistence | Joblib |
| Web application | Streamlit |
| Model development | Kaggle Notebooks |
| Version control | GitHub |
| Deployment | Streamlit Community Cloud |

## 4. Dataset

**Source:** Jigsaw Toxic Comment Classification Challenge

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

The dataset contains 159,571 labeled training comments.

Six classification labels are provided:

- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

Comments may have multiple labels.

**Dataset distribution:**

- Comments without toxicity labels: 143,346
- Comments with at least one toxicity label: 16,225

The dataset is imbalanced, particularly for threats and identity-based hate.

## 5. Machine Learning Architecture

**Input message → Text preprocessing → TF-IDF vectorization → Logistic Regression classifiers → Category scores → Optimized thresholds → Human-review recommendation**

### Text preprocessing

Messages are converted to lowercase, extra whitespace is normalized, and empty records are removed.

### Feature extraction

TF-IDF transforms text into numerical features based on term frequency and relative importance across the training corpus.

The vectorizer uses unigram and bigram features.

### Classification

Six independent Logistic Regression classifiers are trained using a One-vs-Rest strategy.

Each classifier predicts the likelihood of a particular toxicity label.

Class-balanced training weights are used to address label imbalance.

### Threshold optimization

Classification thresholds are selected independently for each category by maximizing F1-score on the validation dataset.

## 6. Dataset Splitting

| Dataset | Records |
|---|---:|
| Training | 127,656 |
| Validation | 15,957 |
| Testing | 15,958 |

The training set is used for model fitting, validation for threshold selection, and testing for final evaluation.

## 7. Model Performance

**Final measured results:**

- Micro F1-score: **0.7330**
- Macro F1-score: **0.6211**

| Category | Precision | Recall | F1 |
|---|---:|---:|---:|
| Toxic | 0.834 | 0.720 | 0.773 |
| Severe Toxic | 0.405 | 0.648 | 0.499 |
| Obscene | 0.827 | 0.761 | 0.792 |
| Threat | 0.444 | 0.541 | 0.488 |
| Insult | 0.704 | 0.770 | 0.735 |
| Identity Hate | 0.384 | 0.514 | 0.440 |

These results were measured on a held-out Jigsaw test split. They are not evidence of equivalent performance on other platforms or languages.

## 8. Optimized Classification Thresholds

| Category | Threshold |
|---|---:|
| Toxic | 0.75 |
| Severe Toxic | 0.85 |
| Obscene | 0.75 |
| Threat | 0.85 |
| Insult | 0.70 |
| Identity Hate | 0.80 |

Thresholds were selected on validation data using F1-score optimization.

## 9. False-Positive and False-Negative Analysis

| Category | False Positives | False Negatives |
|---|---:|---:|
| Toxic | 218 | 426 |
| Severe Toxic | 154 | 57 |
| Obscene | 136 | 205 |
| Threat | 25 | 17 |
| Insult | 262 | 186 |
| Identity Hate | 114 | 67 |

The model misses approximately 46% of labeled threats and 49% of identity-hate comments.

These limitations make human oversight essential.

## 10. Streamlit Application

The deployed application supports:

1. Entering a communication message.
2. Generating category-specific model scores.
3. Displaying labels exceeding their thresholds.
4. Providing a suggested human-review action.

The application does not automatically block, penalize, or enforce actions against users.

## 11. Known Limitations

- Lower recall for rare categories, especially threats and identity hate.
- Potential false positives when messages quote or report abusive language.
- Limited understanding of conversational context.
- Trained primarily on English Wikipedia discussion comments.
- Not validated for multilingual communication or production moderation.
- Model probability scores have not been calibrated.
- Classification thresholds do not necessarily represent optimal human-review thresholds.
- Duplicate and near-duplicate leakage checks across data splits remain to be completed.
- No dedicated phishing or scam classifier is included.

## 12. Future Improvements

- Evaluate separate human-review and classification thresholds.
- Perform duplicate and near-duplicate leakage checks.
- Compare against transformer-based models such as BERT.
- Add more robust contextual and multilingual evaluation.
- Improve rare-category performance.
- Add an independently labeled phishing/scam dataset.
- Introduce confidence calibration and a structured human-review workflow.

## 13. Responsible AI

This application is intended for educational research and decision support.

Model predictions may be incorrect, and potentially harmful messages may go undetected.

Consequential moderation actions should involve appropriately authorized human review.

## 14. Author

Independent Applied AI / NLP portfolio project focused on content moderation, Trust & Safety, and risk decision systems.

**Project Status:** Version 1.0 — Deployed experimental baseline.
