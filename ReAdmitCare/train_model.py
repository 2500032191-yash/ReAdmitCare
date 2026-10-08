import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# =========================================================
# Configuration
# =========================================================

DATA_PATH = "dataset/datasetpreprocessed.csv"
MODEL_PATH = "models"

os.makedirs(MODEL_PATH, exist_ok=True)


# =========================================================
# Load Dataset
# =========================================================

print("=" * 70)
print("ReAdmitCare - Machine Learning")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(df.shape)


# =========================================================
# Separate Features and Target
# =========================================================

X = df.drop("readmitted", axis=1)
y = df["readmitted"]


print("\nTarget distribution:")
print(y.value_counts())


# =========================================================
# Identify Column Types
# =========================================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


print("\nNumber of categorical columns:",
      len(categorical_columns))

print("Number of numerical columns:",
      len(numerical_columns))


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


print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# =========================================================
# Models
# =========================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
}


results = []


# =========================================================
# Train and Evaluate
# =========================================================

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print("Training:", model_name)
    print("=" * 70)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    y_probability = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    # Save model
    filename = model_name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(
        pipeline,
        f"{MODEL_PATH}/{filename}"
    )

    print("Model saved:", f"{MODEL_PATH}/{filename}")


# =========================================================
# Compare Models
# =========================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


# =========================================================
# Save Results
# =========================================================

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nResults saved to:")
print("results/model_comparison.csv")

print("\nMachine Learning completed successfully!")