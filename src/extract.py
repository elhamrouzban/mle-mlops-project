from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from config import cfg


def download_data() -> Path:
    """Download the parquet file to data/raw/ if it is not there yet."""
    cfg.raw_path.parent.mkdir(parents=True, exist_ok=True)

    if not cfg.raw_path.exists():
        urlretrieve(cfg.url, cfg.raw_path)
    return cfg.raw_path


def load_data(data_path: Path) -> pd.DataFrame:
    """Load the parquet file into a DataFrame."""
    return pd.read_parquet(data_path)


if __name__ == "__main__":
    path = download_data()
    df = load_data(path)
