import pandas as pd

INPUT_PATH = "dataset/diabetes.csv"
OUTPUT_PATH = "dataset/datasetpreprocessed.csv"


def preprocess_data():

    print("=" * 60)
    print("ReAdmitCare - Data Preprocessing")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(INPUT_PATH)

    print("\nOriginal dataset shape:")
    print(df.shape)

    # ---------------------------------------------------------
    # 1. Replace '?' with missing values
    # ---------------------------------------------------------
    df = df.replace("?", pd.NA)

    print("\nMissing values after replacing '?':")
    print(df.isnull().sum())

    # ---------------------------------------------------------
    # 2. Remove identifier columns
    # ---------------------------------------------------------
    columns_to_remove = [
        "encounter_id",
        "patient_nbr"
    ]

    df = df.drop(
        columns=[col for col in columns_to_remove if col in df.columns]
    )

    print("\nRemoved identifier columns:")
    print(columns_to_remove)

    # ---------------------------------------------------------
    # 3. Convert readmitted into binary classification target
    # ---------------------------------------------------------
    df["readmitted"] = df["readmitted"].map({
        "<30": 1,
        ">30": 0,
        "NO": 0
    })

    print("\nReadmission distribution:")
    print(df["readmitted"].value_counts())

    # ---------------------------------------------------------
    # 4. Remove columns with too many missing values
    # ---------------------------------------------------------
    high_missing_columns = [
        "weight",
        "payer_code",
        "medical_specialty"
    ]

    df = df.drop(
        columns=[col for col in high_missing_columns if col in df.columns]
    )

    print("\nRemoved high-missing-value columns:")
    print(high_missing_columns)

    # ---------------------------------------------------------
    # 5. Fill categorical missing values
    # ---------------------------------------------------------
    categorical_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in categorical_columns:
        if column != "readmitted":
            df[column] = df[column].fillna("Unknown")

    # ---------------------------------------------------------
    # 6. Fill numerical missing values
    # ---------------------------------------------------------
    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for column in numerical_columns:
        if column != "readmitted":
            df[column] = df[column].fillna(
                df[column].median()
            )

    # ---------------------------------------------------------
    # 7. Save preprocessed dataset
    # ---------------------------------------------------------
    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nFinal dataset shape:")
    print(df.shape)

    print("\nRemaining missing values:")
    print(df.isnull().sum().sum())

    print("\nPreprocessed dataset saved successfully:")
    print(OUTPUT_PATH)

    return df


if __name__ == "__main__":
    preprocess_data()