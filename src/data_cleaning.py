'''
Data Cleaning Class
'''

# Importing necessary libraries
import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Abstract class for handling strategy to handle data
class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


# Data Preprocessing Strategy
class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess Data
        """
        try:
            # Dropping a few columns
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis = 1
            )

            # Filling up the "NULL" Values
            data["product_weight_g"].fillna( data["product_weight_g"].median(), inplace = True)
            data["product_length_cm"].fillna( data["product_length_cm"].median(), inplace = True)
            data["product_height_cm"].fillna( data["product_height_cm"].median(), inplace = True)
            data["product_width_cm"].fillna( data["product_width_cm"].median(), inplace = True)
            data["review_comment_message"].fillna("No review", inplace = True)

            # Dropping columns which contain non-numerical values / Keeping only numerical values' attributes
            data = data.select_dtypes(include = [np.number])

            # Dropping those columns which are not so important
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis = 1)

            return data
        except Exception as e:
            logging.error("Error in preprocessing data : {}".format(e))
            raise e
        

# Strategy for Data Splitting
class DataDivideStrategy(DataStrategy):
    """
    Strategy to divide data into train and test
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data into train and test
        """
        try:
            X = data.drop(["review_score"], axis = 1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test # [pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        except Exception as e:
            logging.error("Error in dividing data : {}", format(e))
            raise e
        

# Data Cleaning Class - to use the above strategies
class DataCleaning:
    """
    Class for cleaning data which preprocesses the data and divides it into train and test
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle Data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data : {}".format(e))
            raise e
        
'''
# If you need to run thie file

if __name__ == "__main__":
    data = pd.read_csv("D:\GitHub_Repos\MLOps-DevOps-for-ML\data\olist_customers_dataset.csv")
    data_cleaning = DataCleaning(data, DataPreProcessStrategy())
    data_cleaning.handle_data()
'''

