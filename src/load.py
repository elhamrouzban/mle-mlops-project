from pathlib import Path

import pandas as pd

from preprocess import RAW_PATH, preprocess

PROCESSED_PATH = Path("./data/processed.parquet")


def save_data(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> Path:
    """Save the transformed DataFrame as a parquet file."""
    path.parent.mkdir(exist_ok=True)
    df.to_parquet(path, index=False)
    return path


if __name__ == "__main__":
    df = pd.read_parquet(RAW_PATH)
    df_processed = preprocess(df)
    path = save_data(df_processed)
    print(f"Saved {df_processed.shape} to {path}")
