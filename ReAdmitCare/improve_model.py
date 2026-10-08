import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# =========================================================
# Configuration
# =========================================================

DATA_PATH = "dataset/datasetpreprocessed.csv"

os.makedirs("models", exist_ok=True)

print("=" * 70)
print("ReAdmitCare - Improved Readmission Model")
print("=" * 70)


# =========================================================
# Load Data
# =========================================================

df = pd.read_csv(DATA_PATH)

X = df.drop("readmitted", axis=1)
y = df["readmitted"]


# =========================================================
# Identify Columns
# =========================================================

categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


print("\nCategorical columns:", len(categorical_columns))
print("Numerical columns:", len(numerical_columns))


# =========================================================
# Preprocessing
# =========================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_columns
        ),
        (
            "cat",
            categorical_transformer,
            categorical_columns
        )
    ]
)


# =========================================================
# Train-Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================================================
# Calculate Class Weight
# =========================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("\nNegative samples:", negative)
print("Positive samples:", positive)
print("Scale positive weight:", round(scale_pos_weight, 2))


# =========================================================
# Balanced XGBoost
# =========================================================

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# =========================================================
# Train
# =========================================================

print("\nTraining improved XGBoost...")

pipeline.fit(X_train, y_train)

print("Training completed.")


# =========================================================
# Prediction Probabilities
# =========================================================

probabilities = pipeline.predict_proba(X_test)[:, 1]


# =========================================================
# Test Different Thresholds
# =========================================================

print("\n" + "=" * 70)
print("THRESHOLD COMPARISON")
print("=" * 70)

thresholds = [0.50, 0.40, 0.30, 0.25, 0.20]

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print(
        f"\nThreshold: {threshold}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )


# =========================================================
# Final Threshold
# =========================================================

FINAL_THRESHOLD = 0.30

final_predictions = (
    probabilities >= FINAL_THRESHOLD
).astype(int)


# =========================================================
# Final Evaluation
# =========================================================

accuracy = accuracy_score(
    y_test,
    final_predictions
)

precision = precision_score(
    y_test,
    final_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    final_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    final_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


print("\n" + "=" * 70)
print("FINAL MODEL RESULTS")
print("=" * 70)

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))


# =========================================================
# Confusion Matrix
# =========================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        final_predictions
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        final_predictions,
        zero_division=0
    )
)


# =========================================================
# Save Improved Model
# =========================================================

joblib.dump(
    pipeline,
    "models/improved_xgboost.pkl"
)

print("\nImproved model saved:")
print("models/improved_xgboost.pkl")

print("\nReAdmitCare improved model completed successfully!")