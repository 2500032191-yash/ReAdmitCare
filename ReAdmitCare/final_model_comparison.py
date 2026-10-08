import pandas as pd
import os

print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

# ---------------------------------------------------------
# CLASSIFICATION RESULTS
# ---------------------------------------------------------

classification_results = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Type": "Classification",
        "Accuracy": 0.6445,
        "Precision": 0.1672,
        "Recall": 0.5491,
        "F1": 0.2563,
        "ROC-AUC": 0.6420
    },
    {
        "Model": "Random Forest",
        "Type": "Classification",
        "Accuracy": 0.8878,
        "Precision": 0.4656,
        "Recall": 0.0387,
        "F1": 0.0715,
        "ROC-AUC": 0.6573
    },
    {
        "Model": "XGBoost",
        "Type": "Classification",
        "Accuracy": 0.8888,
        "Precision": 0.5909,
        "Recall": 0.0114,
        "F1": 0.0225,
        "ROC-AUC": 0.6873
    },
    {
        "Model": "Improved XGBoost",
        "Type": "Classification",
        "Accuracy": 0.0,
        "Precision": 0.1822,
        "Recall": 0.6050,
        "F1": 0.2801,
        "ROC-AUC": 0.0
    },
    {
        "Model": "Decision Tree",
        "Type": "Classification",
        "Accuracy": 0.6112,
        "Precision": 0.1674,
        "Recall": 0.6253,
        "F1": 0.2641,
        "ROC-AUC": 0.6549
    }
])

# ---------------------------------------------------------
# REGRESSION RESULTS
# ---------------------------------------------------------

regression_results = pd.DataFrame([
    {
        "Model": "Random Forest Regression",
        "Type": "Regression",
        "MAE": 1.6890,
        "RMSE": 2.2558,
        "R2": 0.4154
    },
    {
        "Model": "Linear Regression",
        "Type": "Regression",
        "MAE": 1.7113,
        "RMSE": 2.2583,
        "R2": 0.4141
    },
    {
        "Model": "Ridge Regression",
        "Type": "Regression",
        "MAE": 1.7062,
        "RMSE": 2.2512,
        "R2": 0.4178
    },
    {
        "Model": "Lasso Regression",
        "Type": "Regression",
        "MAE": 1.7835,
        "RMSE": 2.3503,
        "R2": 0.3653
    }
])

# ---------------------------------------------------------
# DISPLAY CLASSIFICATION
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CLASSIFICATION MODEL RESULTS")
print("=" * 70)

print(classification_results.to_string(index=False))

# ---------------------------------------------------------
# DISPLAY REGRESSION
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("REGRESSION MODEL RESULTS")
print("=" * 70)

print(regression_results.to_string(index=False))

# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

os.makedirs("results", exist_ok=True)

classification_results.to_csv(
    "results/final_classification_comparison.csv",
    index=False
)

regression_results.to_csv(
    "results/final_regression_comparison.csv",
    index=False
)

# ---------------------------------------------------------
# BEST CLASSIFICATION MODELS
# ---------------------------------------------------------

best_recall = classification_results.loc[
    classification_results["Recall"].idxmax()
]

best_f1 = classification_results.loc[
    classification_results["F1"].idxmax()
]

best_auc = classification_results.loc[
    classification_results["ROC-AUC"].idxmax()
]

print("\n" + "=" * 70)
print("BEST CLASSIFICATION MODELS")
print("=" * 70)

print(
    f"Best Recall : {best_recall['Model']} "
    f"({best_recall['Recall']:.4f})"
)

print(
    f"Best F1     : {best_f1['Model']} "
    f"({best_f1['F1']:.4f})"
)

print(
    f"Best ROC-AUC: {best_auc['Model']} "
    f"({best_auc['ROC-AUC']:.4f})"
)

# ---------------------------------------------------------
# BEST REGRESSION MODELS
# ---------------------------------------------------------

best_mae = regression_results.loc[
    regression_results["MAE"].idxmin()
]

best_rmse = regression_results.loc[
    regression_results["RMSE"].idxmin()
]

best_r2 = regression_results.loc[
    regression_results["R2"].idxmax()
]

print("\n" + "=" * 70)
print("BEST REGRESSION MODELS")
print("=" * 70)

print(
    f"Best MAE : {best_mae['Model']} "
    f"({best_mae['MAE']:.4f})"
)

print(
    f"Best RMSE: {best_rmse['Model']} "
    f"({best_rmse['RMSE']:.4f})"
)

print(
    f"Best R²  : {best_r2['Model']} "
    f"({best_r2['R2']:.4f})"
)

print("\n" + "=" * 70)
print("FINAL COMPARISON COMPLETED")
print("=" * 70)

print("\nFiles created:")
print("results/final_classification_comparison.csv")
print("results/final_regression_comparison.csv")