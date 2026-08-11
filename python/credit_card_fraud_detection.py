"""
Credit Card Fraud Detection
------------------------------
Logistic-regression-based fraud classifier with feature engineering,
under-sampling to handle severe class imbalance, and an ensemble
comparison — matching the approach described in the source resume
project (logistic regression, under-sampling + ensemble techniques,
false-positive minimization).

NOTE ON DATA: A realistic synthetic transaction dataset is generated
below (severely imbalanced, ~1.7% fraud rate, matching real-world
card fraud base rates) with documented relationships between
transaction features and fraud likelihood. This stands in for a real
anonymized transactions dataset. All downstream steps run exactly as
they would on real data with this schema.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)

sns.set_theme(style="whitegrid", palette="deep")
RNG = np.random.default_rng(7)
N = 20000
FRAUD_RATE = 0.017

# ---------------------------------------------------------------------
# 1. DATA GENERATION
# ---------------------------------------------------------------------
n_fraud = int(N * FRAUD_RATE)
n_legit = N - n_fraud

amount_legit = RNG.gamma(2.0, 40, n_legit)
amount_fraud = RNG.gamma(1.2, 220, n_fraud)  # fraud skews to larger/odd amounts

hour_legit = RNG.normal(14, 4.5, n_legit) % 24
hour_fraud = RNG.normal(3, 5, n_fraud) % 24  # fraud skews to odd hours

distance_from_home_legit = np.abs(RNG.normal(5, 8, n_legit))
distance_from_home_fraud = np.abs(RNG.normal(60, 90, n_fraud))

num_transactions_last_hour_legit = RNG.poisson(1.0, n_legit)
num_transactions_last_hour_fraud = RNG.poisson(4.5, n_fraud)

online_order_legit = RNG.choice([0, 1], n_legit, p=[0.65, 0.35])
online_order_fraud = RNG.choice([0, 1], n_fraud, p=[0.15, 0.85])

merchant_risk_score_legit = np.clip(RNG.normal(0.2, 0.15, n_legit), 0, 1)
merchant_risk_score_fraud = np.clip(RNG.normal(0.65, 0.2, n_fraud), 0, 1)

def build(amount, hour, dist, tx_hr, online, risk, label):
    return pd.DataFrame({
        "transaction_amount": amount,
        "hour_of_day": hour,
        "distance_from_home_km": dist,
        "transactions_last_hour": tx_hr,
        "is_online_order": online,
        "merchant_risk_score": risk,
        "is_fraud": label,
    })

df = pd.concat([
    build(amount_legit, hour_legit, distance_from_home_legit,
          num_transactions_last_hour_legit, online_order_legit,
          merchant_risk_score_legit, 0),
    build(amount_fraud, hour_fraud, distance_from_home_fraud,
          num_transactions_last_hour_fraud, online_order_fraud,
          merchant_risk_score_fraud, 1),
], ignore_index=True)
df = df.sample(frac=1, random_state=7).reset_index(drop=True)
df.to_csv("credit_card_transactions.csv", index=False)
print("Shape:", df.shape)
print("Fraud rate:", df["is_fraud"].mean())

# ---------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
df["is_fraud"].value_counts().rename({0: "Legitimate", 1: "Fraud"}).plot(
    kind="bar", color=["#4C72B0", "#C44E52"], ax=ax
)
ax.set_title(f"Class Distribution (Fraud Rate = {df['is_fraud'].mean():.2%})")
ax.set_ylabel("Transactions")
plt.tight_layout()
plt.savefig("images/01_class_imbalance.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df.drop(columns=["is_fraud"]).assign(is_fraud=df["is_fraud"]).corr(),
            annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/02_correlation_heatmap.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x="is_fraud", y="transaction_amount", ax=axes[0])
axes[0].set_title("Transaction Amount by Class")
axes[0].set_xticklabels(["Legitimate", "Fraud"])
sns.boxplot(data=df, x="is_fraud", y="distance_from_home_km", ax=axes[1])
axes[1].set_title("Distance from Home by Class")
axes[1].set_xticklabels(["Legitimate", "Fraud"])
plt.tight_layout()
plt.savefig("images/03_feature_distributions.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# 3. TRAIN/TEST SPLIT + SCALING
# ---------------------------------------------------------------------
feature_cols = ["transaction_amount", "hour_of_day", "distance_from_home_km",
                 "transactions_last_hour", "is_online_order", "merchant_risk_score"]
X = df[feature_cols]
y = df["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------------
# 4. BASELINE: Logistic Regression on imbalanced data
# ---------------------------------------------------------------------
baseline = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
baseline_preds = baseline.predict(X_test_s)

# ---------------------------------------------------------------------
# 5. UNDER-SAMPLING the majority class
# ---------------------------------------------------------------------
train_df = pd.DataFrame(X_train_s, columns=feature_cols)
train_df["is_fraud"] = y_train.values
fraud_train = train_df[train_df.is_fraud == 1]
legit_train = train_df[train_df.is_fraud == 0].sample(
    n=len(fraud_train) * 3, random_state=7  # keep a 3:1 legit:fraud ratio
)
balanced_train = pd.concat([fraud_train, legit_train]).sample(frac=1, random_state=7)
Xb, yb = balanced_train[feature_cols], balanced_train["is_fraud"]

lr_balanced = LogisticRegression(max_iter=1000).fit(Xb, yb)
lr_balanced_preds = lr_balanced.predict(X_test_s)

# ---------------------------------------------------------------------
# 6. ENSEMBLE: Logistic Regression + Random Forest (soft voting), on balanced data
# ---------------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=7, class_weight="balanced")
ensemble = VotingClassifier(
    estimators=[("lr", LogisticRegression(max_iter=1000)), ("rf", rf)],
    voting="soft"
)
ensemble.fit(Xb, yb)
ensemble_preds = ensemble.predict(X_test_s)
ensemble_proba = ensemble.predict_proba(X_test_s)[:, 1]

# ---------------------------------------------------------------------
# 7. COMPARE RESULTS
# ---------------------------------------------------------------------
def score_row(name, y_true, y_pred):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }

results = pd.DataFrame([
    score_row("Logistic Regression (imbalanced)", y_test, baseline_preds),
    score_row("Logistic Regression (under-sampled)", y_test, lr_balanced_preds),
    score_row("Ensemble: LR + Random Forest (under-sampled)", y_test, ensemble_preds),
])
results.to_csv("model_comparison_results.csv", index=False)
print(results)

fig, ax = plt.subplots(figsize=(9, 5))
results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]].plot(kind="bar", ax=ax)
ax.set_title("Model Comparison: Baseline vs Under-Sampled vs Ensemble")
ax.set_ylim(0, 1)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("images/04_model_comparison.png", dpi=150)
plt.close()

# False positive comparison (business-relevant)
fp_baseline = confusion_matrix(y_test, baseline_preds)[0, 1]
fp_ensemble = confusion_matrix(y_test, ensemble_preds)[0, 1]
fp_reduction = (fp_baseline - fp_ensemble) / fp_baseline * 100 if fp_baseline else 0
print(f"False positives baseline: {fp_baseline}, ensemble: {fp_ensemble}, reduction: {fp_reduction:.1f}%")

cm = confusion_matrix(y_test, ensemble_preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"], ax=ax)
ax.set_title("Confusion Matrix — Ensemble Model (Best)")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("images/05_confusion_matrix.png", dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, ensemble_proba)
auc = roc_auc_score(y_test, ensemble_proba)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, label=f"Ensemble (AUC = {auc:.3f})", color="#C44E52", lw=2)
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Ensemble Model")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("images/06_roc_curve.png", dpi=150)
plt.close()

fi = pd.Series(rf.fit(Xb, yb).feature_importances_, index=feature_cols).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
fi.plot(kind="barh", ax=ax, color="#55A868")
ax.set_title("Feature Importance — Random Forest (within ensemble)")
plt.tight_layout()
plt.savefig("images/07_feature_importance.png", dpi=150)
plt.close()

with open("classification_report.txt", "w") as f:
    f.write("=== Baseline (imbalanced) ===\n")
    f.write(classification_report(y_test, baseline_preds))
    f.write("\n=== Under-sampled Logistic Regression ===\n")
    f.write(classification_report(y_test, lr_balanced_preds))
    f.write("\n=== Ensemble (LR + Random Forest, under-sampled) ===\n")
    f.write(classification_report(y_test, ensemble_preds))
    f.write(f"\nFalse positives — baseline: {fp_baseline}, ensemble: {fp_ensemble}, reduction: {fp_reduction:.1f}%\n")
    f.write(f"ROC-AUC (ensemble): {auc:.4f}\n")

print("All artifacts generated successfully.")
