import logging
import pandas as pd
from zenml import step
import numpy as np
from abc import ABC, abstractmethod 
from sklearn.metrics import mean_squared_error, r2_score



class evaluation(ABC):
    @abstractmethod
    def calculate_score(self,y_true:np.ndarray, y_pred:np.ndarray):
        pass

class MSE(evaluation):
    def calculate_score(self,y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("Calculating Mean Squared Error (MSE).")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE calculation completed successfully.")
            return mse
        except Exception as e:
            logging.error(f"Error in MSE calculation: {e}")
            raise e
class R2Score(evaluation):
    def calculate_score(self,y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("Calculating R2 Score.")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 Score calculation completed successfully.")
            return r2
        except Exception as e:
            logging.error(f"Error in R2 Score calculation: {e}")
            raise e
        
class RMSE(evaluation):
    def calculate_score(self,y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("Calculating Root Mean Squared Error (RMSE).")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info("RMSE calculation completed successfully.")
            return rmse
        except Exception as e:
            logging.error(f"Error in RMSE calculation: {e}")
            raise e