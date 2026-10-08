import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

print("=" * 70)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

DATA_PATH = "dataset/datasetpreprocessed.csv"
MODEL_PATH = "models/xgboost.pkl"
OUTPUT_PATH = "static/images/feature_importance.png"

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)

# ---------------------------------------------------------
# SEPARATE TARGET
# ---------------------------------------------------------

X = df.drop(columns=["readmitted"])
y = df["readmitted"]

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

print("\nLoading XGBoost model...")

model_pipeline = joblib.load(MODEL_PATH)

print("XGBoost model loaded successfully.")

# ---------------------------------------------------------
# GET PREPROCESSOR AND MODEL
# ---------------------------------------------------------

preprocessor = model_pipeline.named_steps["preprocessor"]
model = model_pipeline.named_steps["model"]

# ---------------------------------------------------------
# GET FEATURE NAMES
# ---------------------------------------------------------

print("\nExtracting feature names...")

feature_names = preprocessor.get_feature_names_out()

print("Number of processed features:", len(feature_names))

# ---------------------------------------------------------
# GET FEATURE IMPORTANCE
# ---------------------------------------------------------

importance_values = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

# Remove preprocessing prefixes
feature_importance["Feature"] = (
    feature_importance["Feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
)

# Sort by importance
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# ---------------------------------------------------------
# TOP 20 FEATURES
# ---------------------------------------------------------

top_features = feature_importance.head(20)

print("\n" + "=" * 70)
print("TOP 20 IMPORTANT FEATURES")
print("=" * 70)

print(top_features.to_string(index=False))

# ---------------------------------------------------------
# CREATE OUTPUT DIRECTORY
# ---------------------------------------------------------

os.makedirs("static/images", exist_ok=True)

# ---------------------------------------------------------
# CREATE GRAPH
# ---------------------------------------------------------

plt.figure(figsize=(10, 8))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 20 Features for 30-Day Readmission Prediction")

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFeature importance graph saved:")
print(OUTPUT_PATH)

# ---------------------------------------------------------
# SAVE CSV
# ---------------------------------------------------------

CSV_PATH = "results/feature_importance.csv"

os.makedirs("results", exist_ok=True)

feature_importance.to_csv(
    CSV_PATH,
    index=False
)

print("Feature importance data saved:")
print(CSV_PATH)

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE ANALYSIS COMPLETED")
print("=" * 70)