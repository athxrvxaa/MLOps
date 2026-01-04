import logging
import pandas as pd
from zenml import step
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin
import mlflow.sklearn
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> RegressorMixin:

    mlflow.sklearn.autolog()

    try:
        model = LinearRegression()
        model.fit(X_train, y_train)   # 🔑 THIS is what autolog needs
        return model

    except Exception as e:
        logging.error(f"Error in train_model step: {e}")
        raise
