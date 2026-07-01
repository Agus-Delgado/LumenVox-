# LumenVox — Sentiment Modeling Report

## Modeling Objective

The goal of this block is to classify customer feedback sentiment (negative, neutral, or positive) from cleaned text using supervised machine learning models trained on TF-IDF features.

## Business Priority

Negative recall is prioritized because missing negative feedback could delay action on customer pain points. A model that catches more true negatives helps teams respond sooner to issues that drive churn risk.

## Dataset Used

- **Rows used:** 1000
- **Train size:** 800
- **Test size:** 200
- **Target labels:** negative, neutral, positive
- **Text feature:** `clean_message`

## Models Compared

- Logistic Regression
- LinearSVC
- Multinomial Naive Bayes

## Model Comparison

| model_name | accuracy | macro_f1 | weighted_f1 | negative_precision | negative_recall | negative_f1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9550 | 0.9543 | 0.9549 | 0.9412 | 0.9877 | 0.9639 |
| LinearSVC | 0.9800 | 0.9805 | 0.9801 | 0.9529 | 1.0000 | 0.9759 |
| Multinomial Naive Bayes | 0.9150 | 0.9113 | 0.9143 | 0.8989 | 0.9877 | 0.9412 |

## Best Model Selected

**LinearSVC** was selected because it achieved the highest negative recall (1.0000) among the compared models. When models tie on negative recall, macro F1-score is used as the tie-breaker.

## Detailed Classification Reports

### Logistic Regression

```
              precision    recall  f1-score   support

    negative       0.94      0.99      0.96        81
     neutral       0.98      0.95      0.96        56
    positive       0.95      0.92      0.94        63

    accuracy                           0.95       200
   macro avg       0.96      0.95      0.95       200
weighted avg       0.96      0.95      0.95       200
```

### LinearSVC

```
              precision    recall  f1-score   support

    negative       0.95      1.00      0.98        81
     neutral       1.00      0.96      0.98        56
    positive       1.00      0.97      0.98        63

    accuracy                           0.98       200
   macro avg       0.98      0.98      0.98       200
weighted avg       0.98      0.98      0.98       200
```

### Multinomial Naive Bayes

```
              precision    recall  f1-score   support

    negative       0.90      0.99      0.94        81
     neutral       0.98      0.84      0.90        56
    positive       0.89      0.89      0.89        63

    accuracy                           0.92       200
   macro avg       0.92      0.91      0.91       200
weighted avg       0.92      0.92      0.91       200
```

## Interpretation

- Automatically triage incoming feedback by sentiment so support and product teams can focus on negative items first.
- Track negative feedback volume over time as an early indicator of customer satisfaction trends.
- Route high-risk negative messages to escalation workflows before they become churn events.
- Compare model outputs across channels and segments to identify where dissatisfaction is concentrated.
- Provide a reproducible baseline for future topic and product-area classification work.

## Limitations

- The dataset is synthetic and designed for portfolio demonstration.
- Sentiment labels are generated programmatically, not manually annotated by human reviewers.
- Because the dataset is synthetic and label-consistent by design, model performance may be higher than expected on real-world customer feedback.
- Future work should validate model performance against real customer feedback with expert labeling.
- Classical NLP models may miss sarcasm, ambiguity, or context-dependent sentiment.

## Next Steps

- Product area classification using the same text features.
- Topic analysis within negative feedback subsets.
- Error analysis on misclassified feedback to guide improvements.
- Human-in-the-loop validation on a sample of predictions.
- Optional GenAI-assisted executive summaries grounded in validated metrics.

## Generated Artifacts

- `models/sentiment_model.pkl`
- `models/tfidf_vectorizer.pkl`
- `models/model_metadata.json`
- `reports/sentiment_model_comparison.csv`
- `visuals/confusion_matrix_logistic_regression.png`
- `visuals/confusion_matrix_linear_svc.png`
- `visuals/confusion_matrix_naive_bayes.png`
