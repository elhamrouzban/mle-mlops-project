import logging
import os

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

from config import cfg

# Reduce utility log noise without hiding warnings about model packaging.
logging.getLogger("mlflow.utils").setLevel(logging.WARNING)


def setup_mlflow() -> None:
    """Connect to MLflow and create or reuse the experiment."""
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

    if mlflow.get_experiment_by_name(cfg.experiment_name) is None:
        mlflow.create_experiment(
            name=cfg.experiment_name, tags={"developer": "Elham Rouzban"}
        )
    mlflow.set_experiment(cfg.experiment_name)


def split_data(df: pd.DataFrame):
    """Split into 80% train and 20% test."""
    y = df[cfg.target]
    X = df.drop(columns=[cfg.target])
    return train_test_split(X, y, random_state=cfg.random_state, test_size=cfg.test_size)


def register_model(model_info, run_id: str) -> int:
    """Register the model and give the new version the production alias."""
    client = MlflowClient()

    try:
        client.create_registered_model(
            name=cfg.model_name,
            tags={"framework": "scikit-learn"},
            description="Model to predict ride duration in NYC",
        )
    except MlflowException as exc:
        # Reuse the model only if it already exists; re-raise any other MLflow error.
        if exc.error_code != "RESOURCE_ALREADY_EXISTS":
            raise

    mv = client.create_model_version(
        name=cfg.model_name,
        source=model_info.model_uri,
        model_id=model_info.model_id,
        run_id=run_id,
    )
    client.set_registered_model_alias(
        name=cfg.model_name, version=mv.version, alias="production"
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
                "dataset": f"{cfg.color}-taxi",
                "year": cfg.year,
                "month": cfg.month,
                "features": cfg.features,
                "target": cfg.target,
            }
        )
        mlflow.log_input(
            mlflow.data.from_pandas(df, source=str(cfg.processed_path)),
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
    print(f"Registered model: {cfg.model_name}, version {version}, alias: production")


if __name__ == "__main__":
    setup_mlflow()
    df = pd.read_parquet(cfg.processed_path)
    train(df)
