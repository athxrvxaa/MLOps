import logging
import pandas as pd
from src.eval import MSE, R2Score, RMSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from zenml.client import Client
import mlflow

from zenml import step

ecperiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=ecperiment_tracker.name)
def evaluate_model(model:RegressorMixin,
                X_test:pd.DataFrame,
                Y_test:pd.Series) -> Tuple[
                    Annotated[float, "R2 Score"],
                    Annotated[float, "RMSE"]]:
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_score(Y_test, prediction)
        mlflow.log_metric("MSE", mse)

        r2_class = R2Score()
        r2 = r2_class.calculate_score(Y_test, prediction)
        mlflow.log_metric("R2_Score", r2)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(Y_test, prediction)
        logging.info(f"Model Evaluation Results: MSE={mse}, R2={r2}, RMSE={rmse}")
        mlflow.log_metric("RMSE", rmse)

        return r2 , rmse
    except Exception as e:
        logging.error(f"Error in evaluate_model: {e}")
        raise e