from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "params.yaml"


@dataclass(frozen=True)
class Config:
    year: int
    month: int
    color: str
    raw_dir: Path
    processed_dir: Path
    features: list[str]
    target: str
    test_size: float
    random_state: int
    experiment_name: str
    model_name: str

    @property
    def file_name(self) -> str:
        return f"{self.color}_tripdata_{self.year}-{self.month:02d}.parquet"

    @property
    def url(self) -> str:
        return f"https://d37ci6vzurychx.cloudfront.net/trip-data/{self.file_name}"

    @property
    def raw_path(self) -> Path:
        return self.raw_dir / self.file_name

    @property
    def processed_path(self) -> Path:
        return self.processed_dir / self.file_name


def load_config(path: Path = PARAMS_PATH) -> Config:
    """Read params.yaml and return a Config object."""
    with open(path) as f:
        params = yaml.safe_load(f)

    data = params["data"]
    return Config(
        year=data["year"],
        month=data["month"],
        color=data["color"],
        raw_dir=ROOT / data["raw_dir"],
        processed_dir=ROOT / data["processed_dir"],
        features=params["features"],
        target=params["target"],
        test_size=params["train"]["test_size"],
        random_state=params["train"]["random_state"],
        experiment_name=params["mlflow"]["experiment_name"],
        model_name=params["mlflow"]["model_name"],
    )


cfg = load_config()
