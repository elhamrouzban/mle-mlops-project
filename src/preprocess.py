import pandas as pd

from config import cfg


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate duration, remove outliers, and treat the location IDs as categories."""
    df = df.copy()

    df[cfg.target] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df[cfg.target] >= 1) & (df[cfg.target] <= 60)]
    df = df[df["trip_distance"].between(0.1, 100)]

    df[["PULocationID", "DOLocationID"]] = df[["PULocationID", "DOLocationID"]].astype(
        str
    )
    return df[cfg.features + [cfg.target]]


if __name__ == "__main__":
    df = pd.read_parquet(cfg.raw_path)
    df_processed = preprocess(df)
    print(df_processed.shape)
