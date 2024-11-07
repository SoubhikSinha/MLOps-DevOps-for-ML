'''
Class(es) for Model Deployment (Local Machine)
'''

# importing necessary libraries
import numpy as np
import pandas as pd
import mlflow
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from pydantic import BaseModel

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model


# Docker Settings
docker_settings = DockerSettings(required_integrations=[MLFLOW])

# Before model deployment, it should meet some minimum criteria - thus, we need to have a class for that
class DeploymentTriggerConfig(BaseModel):
    """
    Deployment Trigger Config with minimum accuracy requirement
    """
    min_accuracy: float = 0.92

    class Config:
        protected_namespaces = ()  # Fixing Pydantic warning

@step
def deployment_trigger(accuracy: float, config: DeploymentTriggerConfig):
    """
    Model deployment trigger that returns True if model accuracy meets or exceeds the min_accuracy
    """
    return accuracy >= config.min_accuracy

# Continuous Deployment Pipeline (CD Pipeline)
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuos_deployment_pipeline(data_path: str, min_accuracy: float = 0.92, workers: int = 1, timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT):
    """
    Pipeline for continuous deployment that triggers deployment only if model accuracy meets minimum threshold
    """
    # Setting up or create the experiment
    experiment_name = "continuos_deployment_pipeline"
    experiment_id = mlflow.get_experiment_by_name(experiment_name)
    if not experiment_id:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Pipeline steps
    df = ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2, rmse = evaluate_model(model, X_test, y_test)

    # Deployment Trigger
    config = DeploymentTriggerConfig(min_accuracy=min_accuracy)
    deployment_decision = deployment_trigger(accuracy=r2, config=config)

    # Conditional model deployment based on deployment_decision
    if deployment_decision:
        mlflow_model_deployer_step(
            model=model,
            deploy_decision=deployment_decision,
            workers=workers,
            timeout=timeout,
        )
