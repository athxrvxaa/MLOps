import logging
import pandas as pd
from zenml import step

from typing_extensions import Annotated
from typing import Tuple
from src.data_cleaning import DataCleaning, DataPreprocessorStrategy, DataSplitterStrategy 

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"], 
]:
    try:
        process_strategy = DataPreprocessorStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        data_divide_strategy = DataSplitterStrategy()
        data_cleaning = DataCleaning(processed_data, data_divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()  
        logging.info("Data cleaning and splitting completed successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in clean_df during preprocessing: {e}")
        raise e 

