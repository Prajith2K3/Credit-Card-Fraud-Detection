[README.md](https://github.com/user-attachments/files/30933679/README.md)
# Credit Card Fraud Detection

Logistic-regression fraud classifier with feature engineering and under-sampling to handle severe class imbalance, benchmarked against an ensemble model.

> **Data note:** Rebuilt on a documented synthetic dataset — 20,000 transactions at a 1.7% fraud rate (matching real-world card fraud base rates), with fraud transactions skewed toward odd hours, larger/atypical amounts, unusual distance-from-home, and higher merchant risk scores. Stands in for a real anonymized transactions dataset; pipeline runs identically on real data with this schema.

## Business Problem
Card issuers need to catch fraudulent transactions in real time while keeping false declines (blocking legitimate customers) low.

## Dataset
- 20,000 transactions, 6 features, severely imbalanced target (`is_fraud`)
- Features: `transaction_amount`, `hour_of_day`, `distance_from_home_km`, `transactions_last_hour`, `is_online_order`, `merchant_risk_score`

## Pipeline
1. **EDA** — class imbalance chart, correlation heatmap, amount/distance distributions by class
2. **Baseline** — Logistic Regression trained directly on the imbalanced data
3. **Under-sampling** — majority (legitimate) class downsampled to a 3:1 ratio against fraud cases, retrained
4. **Ensemble** — soft-voting Logistic Regression + Random Forest on the under-sampled data
5. **Evaluation** — accuracy, precision, recall, F1, confusion matrix, ROC-AUC, feature importance

## Results (this run)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (imbalanced) | 0.9992 | 1.000 | 0.953 | 0.976 |
| Logistic Regression (under-sampled) | 0.9968 | 0.848 | 0.988 | 0.913 |
| **Ensemble: LR + Random Forest (under-sampled)** | 0.9978 | 0.894 | **0.988** | 0.939 |

ROC-AUC (ensemble): **0.9999**

**Honest trade-off observed:** under-sampling raised fraud recall from 95.3% → 98.8% (catches more real fraud) at the cost of precision (more false positives) versus the imbalanced baseline. The ensemble model recovered some of that lost precision while keeping the higher recall — the standard precision/recall trade-off you'd expect when rebalancing a fraud dataset. Full breakdown in `classification_report.txt`.

## Files in this package
```
credit_card_fraud_detection/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── data/
│   ├── credit_card_transactions.csv
│   ├── model_comparison_results.csv
├── docs/
│   ├── classification_report.txt
├── python/
│   ├── Credit_Card_Fraud_Detection.ipynb
│   ├── credit_card_fraud_detection.py
└── images/
    ├── 01_class_imbalance.png
    ├── 02_correlation_heatmap.png
    ├── 03_feature_distributions.png
    ├── 04_model_comparison.png
    ├── 05_confusion_matrix.png
    ├── 06_roc_curve.png
    └── 07_feature_importance.png
```

## Suggested LinkedIn caption

> 💳 **Credit Card Fraud Detection — Handling Extreme Class Imbalance**
>
> Built a fraud detection pipeline on a 20K-transaction dataset with a realistic 1.7% fraud rate.
>
> 🔧 What I did:
> • Engineered features around timing, distance-from-home, and merchant risk
> • Compared a baseline Logistic Regression against an under-sampled + ensemble approach
> • Improved fraud recall from 95.3% → 98.8% by rebalancing the training data
> • Evaluated with ROC-AUC (0.9999), confusion matrices, and full precision/recall trade-off analysis
>
> 🛠️ Tools: Python, Scikit-Learn, Pandas, Matplotlib, Seaborn
>
> #DataScience #MachineLearning #FraudDetection #Python #Portfolio

## Limitations
- Synthetic data with clean, documented separability — real transaction data is noisier and harder to separate
- Under-sampling discards majority-class data; SMOTE or cost-sensitive learning are worth comparing on real data
- Not deployed or tested against real-time transaction streams
