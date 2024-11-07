'''
Model Development Class
'''

# Importing necessary libraries
import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract Class for all models
    """
    
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        Args:
            X_train : Training Data
            y_train : Training Labels
        Returns:
            None
        """
        pass


class LinearRegressionModel(Model):
    """
    Linear Regression Model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train : Training Data
            y_train : Training Labels
        Returns:
            None 
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error("Error in Training Model : {}".format(e))
            raise e