import logging 
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self,data: pd.DataFrame) -> Union[pd.DataFrame , pd.Series]:
        pass

class DataPreprocessorStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            date_columns = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
          ]
            existing_date_columns = [col for col in date_columns if col in data.columns]

            data = data.dropna(subset=existing_date_columns)




            data.loc[:, "product_weight_g"] = data["product_weight_g"].fillna(
                data["product_weight_g"].median()
            )

            data.loc[:, "product_length_cm"] = data["product_length_cm"].fillna(
                data["product_length_cm"].median()
            )

            data.loc[:, "product_height_cm"] = data["product_height_cm"].fillna(
                data["product_height_cm"].median()
            )

            data.loc[:, "product_width_cm"] = data["product_width_cm"].fillna(
                data["product_width_cm"].median()
            )

            data.loc[:, "review_comment_message"] = data["review_comment_message"].fillna(
                "No comment"
            )

        
            data =data.select_dtypes(include=[np.number])
            cols_to_drop = ["order_id", "customer_zip_code_prefix"]
            data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])

            return data
        except Exception as e:
            logging.error(f"Error in DataPreprocessorStrategy: {e}")
            raise

class DataSplitterStrategy(DataStrategy):
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame , pd.Series]:
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test =train_test_split(
                X,y,test_size = 0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in DataSplitterStrategy: {e}")
            raise e


class DataCleaning:
        def __init__(self,data: pd.DataFrame, strategy: DataStrategy):
            self.strategy = strategy
            self.data = data
        def handle_data(self) -> Union[pd.DataFrame , pd.Series]:
            try:
                return self.strategy.handle_data(self.data)
            except Exception as e:
                logging.error(f"Error in DataCleaning: {e}")
                raise e
            

if __name__ == "__main__":
    data = pd.read_csv("E:\MLOps\mlops-mine\data\olist_customers_dataset.csv")
    data_cleaning = DataCleaning(data, DataPreprocessorStrategy())
    data_cleaning.handle_data()