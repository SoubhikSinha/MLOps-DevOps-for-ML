'''
Model Evaluation
'''

# Importing necessary libraries
import logging
import pandas as pd
from zenml import step
from zenml.client import Client
import mlflow

from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated

from src.evaluation import MSE, RMSE, R2

experiment_tracker = Client().active_stack.experiment_tracker

# Method for model evaluation
@step(experiment_tracker = experiment_tracker.name)
def evaluate_model(
    model : RegressorMixin,
    X_test : pd.DataFrame,
    y_test : pd.DataFrame
) -> Tuple[
    Annotated[float, "r2"],
    Annotated[float, "rmse"],
]:
    """
    Evaluates the model on the ingested data.
    Args:
        df: the ingested data
    """
    try:  
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("mse", mse)

        r2_class = R2()
        r2 = r2_class.calculate_score(y_test, prediction)
        mlflow.log_metric("r2", r2)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("rmse", rmse)

        return r2, rmse
    except Exception as e:
        logging.error("Error in Evaluating Model : {}".format(e))
        raise e