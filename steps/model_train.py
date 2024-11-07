'''
Model Training
'''

# importing necessary libraries
import logging
import pandas as pd
from zenml import step

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker
'''
👆 enables you to systematically compare the outcomes of different 
models or experiments side by side and helps in identifying which 
one worked the best for you.
'''

# Method for training the model
@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train : pd.DataFrame,
    X_test : pd.DataFrame,
    y_train : pd.Series,
    y_test : pd.Series,
    config : ModelNameConfig = ModelNameConfig(),  # Using default config if none is provided,
) -> RegressorMixin: # This is how we will return the Linear Regression Model
    """
    Trains the model on the ingested data.

    Args:
        X_train : pd.DataFrame,
        X_test : pd.DataFrame,
        y_train : pd.Series,
        y_test : pd.Series
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            # We need to Log ou model (Linear Regression Model)
            mlflow.sklearn.autolog() # Automatically log your model, scores, etc.
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error("Error in training model : {}".format(e))
        raise e