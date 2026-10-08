import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)

os.makedirs("static/images", exist_ok=True)

print("=" * 70)
print("ReAdmitCare - Model Evaluation")
print("=" * 70)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv("dataset/datasetpreprocessed.csv")

# ---------------------------------------------------------
# CLASSIFICATION DATA
# ---------------------------------------------------------

X = df.drop(columns=["readmitted"])
y = df["readmitted"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# LOAD XGBOOST MODEL
# ---------------------------------------------------------

model_path = "models/xgboost.pkl"

if os.path.exists(model_path):

    model = joblib.load(model_path)

    print("\nLoaded XGBoost model.")

    y_probability = model.predict_proba(X_test)[:, 1]

    y_prediction = (y_probability >= 0.5).astype(int)

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_prediction
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Readmitted", "Readmitted"],
        yticklabels=["Not Readmitted", "Readmitted"]
    )

    plt.title("Readmission Prediction - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        "static/images/confusion_matrix.png"
    )

    plt.close()

    print("Saved: confusion_matrix.png")

    # -----------------------------------------------------
    # ROC CURVE
    # -----------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_test,
        y_probability
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"XGBoost (AUC = {roc_auc:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve - Readmission Prediction")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "static/images/roc_curve.png"
    )

    plt.close()

    print("Saved: roc_curve.png")

else:

    print(
        "\nXGBoost model not found."
    )


# ---------------------------------------------------------
# REGRESSION EVALUATION
# ---------------------------------------------------------

regression_model = joblib.load(
    "models/hospital_stay_regressor.pkl"
)

target = "time_in_hospital"

X_reg = df.drop(
    columns=[
        target,
        "readmitted"
    ]
)

y_reg = df[target]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

y_pred_reg = regression_model.predict(
    X_test_reg
)

# ---------------------------------------------------------
# ACTUAL VS PREDICTED
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.scatter(
    y_test_reg,
    y_pred_reg,
    alpha=0.3
)

plt.plot(
    [1, 14],
    [1, 14],
    linestyle="--"
)

plt.xlabel("Actual Hospital Stay")
plt.ylabel("Predicted Hospital Stay")

plt.title(
    "Actual vs Predicted Hospital Stay"
)

plt.tight_layout()

plt.savefig(
    "static/images/actual_vs_predicted.png"
)

plt.close()

print(
    "Saved: actual_vs_predicted.png"
)

# ---------------------------------------------------------
# REGRESSION ERROR PLOT
# ---------------------------------------------------------

errors = y_test_reg - y_pred_reg

plt.figure(figsize=(7, 5))

plt.hist(
    errors,
    bins=30
)

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")

plt.title(
    "Hospital Stay Prediction Error Distribution"
)

plt.tight_layout()

plt.savefig(
    "static/images/regression_errors.png"
)

plt.close()

print(
    "Saved: regression_errors.png"
)

print("\n" + "=" * 70)
print("MODEL EVALUATION COMPLETED SUCCESSFULLY!")
print("=" * 70)