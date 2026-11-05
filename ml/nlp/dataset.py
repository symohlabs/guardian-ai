from pathlib import Path

import pandas as pd


# Locate the dataset relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "dataset.csv"

def load_dataset()->pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH, sep=",", quotechar='"', skipinitialspace=True)

    print("CSV path:", DATASET_PATH)
    print("Raw columns:", repr(df.columns.tolist()))
    print("First row:", df.head(1).to_dict())

    # Clean column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace("\ufeff", "", regex=False)
    )

    print("Cleaned columns:", repr(df.columns.tolist()))

    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Expected columns {required_columns}, "
            f"but pandas detected {set(df.columns)}"
        )

    df = df.dropna(subset=["text", "label"])
    df = df.drop_duplicates(subset=["text", "label"])

    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    return df

if __name__ == "__main__":
    dataset = load_dataset()

    print("\nDataset loaded successfully!")
    print(f"Total examples: {len(dataset)}")

    print("\nExamples per category:")
    print(dataset["label"].value_counts())

    print("\nFirst five examples:")
    print(dataset.head())