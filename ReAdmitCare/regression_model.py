import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# Configuration
# =========================================================

DATA_PATH = "dataset/datasetpreprocessed.csv"

os.makedirs("models", exist_ok=True)

print("=" * 70)
print("ReAdmitCare - Hospital Stay Regression")
print("=" * 70)


# =========================================================
# Load Dataset
# =========================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(df.shape)


# =========================================================
# Target
# =========================================================

target = "time_in_hospital"

X = df.drop(columns=[target, "readmitted"])

y = df[target]


print("\nTarget:", target)

print("Target statistics:")
print(y.describe())


# =========================================================
# Column Types
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
# Train/Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# =========================================================
# Random Forest Regressor
# =========================================================

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
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

print("\nTraining Random Forest Regressor...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# =========================================================
# Prediction
# =========================================================

y_pred = pipeline.predict(X_test)


# =========================================================
# Metrics
# =========================================================

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


print("\n" + "=" * 70)
print("REGRESSION RESULTS")
print("=" * 70)

print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R²  :", round(r2, 4))


# =========================================================
# Example Predictions
# =========================================================

print("\nExample predictions:")

comparison = pd.DataFrame({
    "Actual": y_test.iloc[:10].values,
    "Predicted": np.round(
        y_pred[:10],
        2
    )
})

print(comparison)


# =========================================================
# Save Model
# =========================================================

joblib.dump(
    pipeline,
    "models/hospital_stay_regressor.pkl"
)

print("\nModel saved:")
print("models/hospital_stay_regressor.pkl")

print("\nRegression completed successfully!")