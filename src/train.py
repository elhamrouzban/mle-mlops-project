import logging
import os
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from preprocess import FEATURES, TARGET

# Reduce utility log noise without hiding warnings about model packaging.
logging.getLogger("mlflow.utils").setLevel(logging.WARNING)

PROCESSED_PATH = Path("./data/processed/green_tripdata_2024-01.parquet")
EXPERIMENT_NAME = "green-taxi-monitoring"
MODEL_NAME = "green-taxi-ride-duration"
YEAR = 2024
MONTH = 1
COLOR = "green"


def setup_mlflow() -> None:
    """Connect to MLflow and create or reuse the experiment."""
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            name=EXPERIMENT_NAME, tags={"developer": "Elham Rouzban"}
        )
    mlflow.set_experiment(EXPERIMENT_NAME)


def split_data(df: pd.DataFrame):
    """Split into 80% train and 20% test."""
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return train_test_split(X, y, random_state=42, test_size=0.2)


def register_model(model_info, run_id: str) -> int:
    """Register the model and give the new version the production alias."""
    client = MlflowClient()

    try:
        client.create_registered_model(
            name=MODEL_NAME,
            tags={"framework": "scikit-learn"},
            description="Model to predict ride duration in NYC",
        )
    except MlflowException as exc:
        # Reuse the model only if it already exists; re-raise any other MLflow error.
        if exc.error_code != "RESOURCE_ALREADY_EXISTS":
            raise

    mv = client.create_model_version(
        name=MODEL_NAME,
        source=model_info.model_uri,
        model_id=model_info.model_id,
        run_id=run_id,
    )
    client.set_registered_model_alias(
        name=MODEL_NAME, version=mv.version, alias="production"
    )
    return mv.version


def train(df: pd.DataFrame) -> None:
    """Train the model, log it to MLflow, and register it."""
    X_train, X_test, y_train, y_test = split_data(df)

    with mlflow.start_run() as run:
        mlflow.set_tags(
            {
                "model": "linear regression",
                "developer": "Elham Rouzban",
                "dataset": f"{COLOR}-taxi",
                "year": YEAR,
                "month": MONTH,
                "features": FEATURES,
                "target": TARGET,
            }
        )
        mlflow.log_input(
            mlflow.data.from_pandas(df, source=str(PROCESSED_PATH)),
            context="training",
        )

        pipeline = make_pipeline(DictVectorizer(), LinearRegression())
        pipeline.fit(X_train.to_dict(orient="records"), y_train)

        y_pred = pipeline.predict(X_test.to_dict(orient="records"))
        rmse = root_mean_squared_error(y_test, y_pred)
        mlflow.log_metric("rmse", rmse)

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline, name="mlflow-model", serialization_format="skops"
        )
        version = register_model(model_info, run.info.run_id)

    print(f"RMSE: {rmse}")
    print(f"Registered model: {MODEL_NAME}, version {version}, alias: production")


if __name__ == "__main__":
    setup_mlflow()
    df = pd.read_parquet(PROCESSED_PATH)
    train(df)
