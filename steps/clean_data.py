'''
Data Cleaning
'''

# Importing Libraries
import logging
import pandas as pd
from zenml import step

from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy
from typing_extensions import Annotated
from typing import Tuple

# Method for cleaning data
@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"], # pd.DataFrame
    Annotated[pd.DataFrame, "X_test"], # pd.DataFrame
    Annotated[pd.Series, "y_train"], # pd.Series
    Annotated[pd.Series, "y_test"], # pd.Series
]:
    """
    Cleans the data and divides it into train and test

    Args:
        df: Raw Data
    Returns:
        X_train : Training Data
        X_test : Testing Data
        y_train : Training Labels
        y_test : Testing Labels
    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()

        return X_train, X_test, y_train, y_test
        
        logging.info("Data Cleaning Completed")
    except Exception as e:
        logging.error("Error in cleaning data : {}".format(e))
        raise e