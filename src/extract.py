from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

YEAR = 2024
MONTH = 1
COLOR = "green"


def download_data(year: int = YEAR, month: int = MONTH, color: str = COLOR) -> Path:
    """Download the parquet file to data/ if it is not there yet."""

    data_path = Path(f"./data/raw/{color}_tripdata_{year}-{month:02d}.parquet")
    data_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        urlretrieve(
            f"https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month:02d}.parquet",
            data_path,
        )
    return data_path


def load_data(data_path: Path) -> pd.DataFrame:
    """Load the parquet file into a DataFrame."""
    return pd.read_parquet(data_path)


if __name__ == "__main__":
    path = download_data()
    df = load_data(path)
