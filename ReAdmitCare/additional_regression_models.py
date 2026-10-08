import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("ADDITIONAL REGRESSION MODELS")
print("=" * 70)

df = pd.read_csv(
    "dataset/datasetpreprocessed.csv"
)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# 2. TARGET = TIME IN HOSPITAL
# ============================================================

X = df.drop(
    ["time_in_hospital", "readmitted"],
    axis=1
)

y = df["time_in_hospital"]

print("\nTarget: time_in_hospital")

print("\nTarget statistics:")
print(y.describe())


# ============================================================
# 3. COLUMN TYPES
# ============================================================

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("\nNumerical columns:", len(numerical_columns))
print("Categorical columns:", len(categorical_columns))


# ============================================================
# 4. PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

preprocessor = ColumnTransformer([
    (
        "num",
        numerical_pipeline,
        numerical_columns
    ),
    (
        "cat",
        categorical_pipeline,
        categorical_columns
    )
])


# ============================================================
# 5. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Ridge Regression":
        Ridge(
            alpha=1.0
        ),

    "Lasso Regression":
        Lasso(
            alpha=0.01,
            max_iter=5000
        )
}


# ============================================================
# 7. CREATE FOLDERS
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# 8. TRAIN MODELS
# ============================================================

results = []

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print("Training:", model_name)
    print("=" * 70)

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    # Predictions
    y_pred = pipeline.predict(
        X_test
    )

    # Metrics
    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    print(
        "MAE :",
        round(mae, 4)
    )

    print(
        "RMSE:",
        round(rmse, 4)
    )

    print(
        "R²  :",
        round(r2, 4)
    )

    # Save model
    model_filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + ".pkl"
    )

    model_path = os.path.join(
        "models",
        model_filename
    )

    joblib.dump(
        pipeline,
        model_path
    )

    print(
        "Saved:",
        model_path
    )

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# ============================================================
# 9. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_path = (
    "results/additional_regression_models.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL REGRESSION RESULTS")
print("=" * 70)

print(
    results_df.round(4).to_string(
        index=False
    )
)

print("\nResults saved to:")
print(results_path)

print("\n" + "=" * 70)
print("ALL REGRESSION MODELS COMPLETED")
print("=" * 70)