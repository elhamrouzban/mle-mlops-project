from pathlib import Path

import pandas as pd

from config import cfg
from preprocess import preprocess


def save_data(df: pd.DataFrame, path: Path = cfg.processed_path) -> Path:
    """Save the transformed DataFrame as a parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


if __name__ == "__main__":
    df = pd.read_parquet(cfg.raw_path)
    df_processed = preprocess(df)
    path = save_data(df_processed)
    print(f"Saved {df_processed.shape} to {path}")
