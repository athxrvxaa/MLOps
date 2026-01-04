import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from pydantic import BaseModel

from steps.clean_data import clean_df
from steps.ecaluation import evaluate_model
from steps.ingest_data import ingest_data
from steps.model_train import train_model


DockerSettings = DockerSettings(
    required_integrations=[MLFLOW]
)

class DeploymentTriggerConfig(BaseModel):
    min_accuracy: float = 0.92

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
):
    return accuracy >= config.min_accuracy


@pipeline(enable_cache=False, settings={"docker": DockerSettings})
def continuous_deployment_pipeline(
    data_path: str = "/home/atharva/MLOps/mlops-mine/data/olist_customers_dataset.csv",
    min_accuracy: float = 0.92,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_data(
        data_path="/home/atharva/MLOps/mlops-mine/data/olist_customers_dataset.csv"
    )

    X_train, X_test, y_train, y_test = clean_df(df)

    model = train_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    r2, rmse = evaluate_model(model, X_test, y_test)

    deployment_decision = deployment_trigger(
        accuracy=r2,
        config=DeploymentTriggerConfig(min_accuracy=min_accuracy),
    )

    mlflow_model_deployer_step(
        model,
        deployment_decision,
        workers=workers,
        timeout=timeout,
    )



@pipeline(enable_cache=False)
def inference_pipeline(
    pipeline_name: str,
    pipeline_step_name: str,
):
    """
    Fetches a deployed MLflow model and keeps the service alive.
    """
    mlflow_model_deployer_step(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
    )
