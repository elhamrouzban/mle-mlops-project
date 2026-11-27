from pathlib import Path

import pandas as pd

RAW_PATH = Path("./data/green_tripdata_2024-01.parquet")

FEATURES = ["PULocationID", "DOLocationID", "trip_distance"]
TARGET = "duration"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate duration, remove outliers, and treat the location IDs as categories."""
    df = df.copy()

    df[TARGET] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df[TARGET] >= 1) & (df[TARGET] <= 60)]
    df = df[df["trip_distance"].between(0.1, 100)]

    df[["PULocationID", "DOLocationID"]] = df[["PULocationID", "DOLocationID"]].astype(
        str
    )
    return df[FEATURES + [TARGET]]


if __name__ == "__main__":
    df = pd.read_parquet(RAW_PATH)
    df_processed = preprocess(df)
    print(df_processed.shape)
