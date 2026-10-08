import pandas as pd

DATA_PATH = "dataset/diabetes.csv"


def load_dataset():
    df = pd.read_csv(DATA_PATH)

    print("=" * 50)
    print("ReAdmitCare Dataset")
    print("=" * 50)

    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nFirst 5 records:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    return df


if __name__ == "__main__":
    load_dataset()