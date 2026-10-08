import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from lightgbm import LGBMClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("FAST ADDITIONAL CLASSIFICATION MODELS")
print("=" * 70)

df = pd.read_csv("dataset/datasetpreprocessed.csv")

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

X = df.drop("readmitted", axis=1)
y = df["readmitted"]

print("\nTarget distribution:")
print(y.value_counts())


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
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])


# ============================================================
# 5. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. PREPROCESS ONLY ONCE
# ============================================================

print("\nPreprocessing data...")

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Preprocessing completed.")

print(
    "Processed training shape:",
    X_train_processed.shape
)


# ============================================================
# 7. MODELS
# ============================================================

models = {

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        class_weight="balanced",
        random_state=42
    ),

    "AdaBoost": AdaBoostClassifier(
        n_estimators=50,
        learning_rate=0.5,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
}


# ============================================================
# 8. CREATE FOLDERS
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# 9. TRAIN MODELS
# ============================================================

results = []

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print("Training:", model_name)
    print("=" * 70)

    model.fit(
        X_train_processed,
        y_train
    )

    y_pred = model.predict(
        X_test_processed
    )

    y_probability = model.predict_proba(
        X_test_processed
    )[:, 1]

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

    # Save model + preprocessor together
    complete_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    model_filename = (
        model_name.lower()
        .replace(" ", "_")
        + ".pkl"
    )

    model_path = os.path.join(
        "models",
        model_filename
    )

    joblib.dump(
        complete_model,
        model_path
    )

    print("Saved:", model_path)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })


# ============================================================
# 10. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_path = (
    "results/additional_classification_models.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# 11. DISPLAY FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    results_df.round(4).to_string(
        index=False
    )
)

print("\nResults saved to:")
print(results_path)

print("\n" + "=" * 70)
print("ALL ADDITIONAL MODELS COMPLETED SUCCESSFULLY")
print("=" * 70)