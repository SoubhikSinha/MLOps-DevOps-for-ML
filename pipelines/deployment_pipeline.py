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

from pipelines.utils import get_data_for_test
import json


# Docker Settings
docker_settings = DockerSettings(required_integrations=[MLFLOW])

# Before model deployment, it should meet some minimum criteria - thus, we need to have a class for that
class DeploymentTriggerConfig(BaseModel):
    """
    Deployment Trigger Config with minimum accuracy requirement
    """
    min_accuracy: float = 0

    class Config:
        protected_namespaces = ()  # Fixing Pydantic warning

@step(enable_cache=False)
def dynamic_importer() -> str:
    """
    Getting Data for Testing
    """
    data = get_data_for_test()
    return data


@step
def deployment_trigger(accuracy: float, config: DeploymentTriggerConfig):
    """
    Model deployment trigger that returns True if model accuracy meets or exceeds the min_accuracy
    """
    return accuracy > config.min_accuracy

class MLFlowDeploymentLoaderStepParameters(BaseModel):
    """
    MLFLow deployment getter parameters

    Attributes:
        pipeline_name : Name of the pipeline that deployed the MLflow prediction server
        step_name : The Name of the step that deployed the MLflow prediction server
        running : When this flag is set, the step only returs a running service
        model_name : The name of the model that is deployed
    """
    pipeline_name : str
    step_name : str
    running : bool = True

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name : str,
    pipeline_step_name : str,
    running : bool = True,
    model_name : str = "model",
) ->MLFlowDeploymentService:
    """
    Get the prediction service started by the deployment pipeline

    Args:
        pipeline_name : Name of the pipeline that deployed the MLflow prediction server
        pipeline_step_name : The Name of the step that deployed the MLflow prediction server
        running : When this flag is set, the step only returs a running service
        model_name : The name of the model that is deployed
    """

    # Getting the MLflow deployment tstack component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()

    # Fetching existing services with same pipeline name, step name and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name = pipeline_step_name,
        model_name=model_name,
        running = running
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow deployment service found for pipeline {pipeline_name}, "
            f"step {pipeline_step_name} and model {model_name}."
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    
    return existing_services[0]

@step
def predictor( # Prediction Service
    service : MLFlowDeploymentService,
    data : str,
) -> np.array:
    service.start(timeout=10)
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm"
    ]
    df = pd.DataFrame(data["data"], columns = columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.ndarray(json_list)
    prediction = service.predict(data)
    return prediction

# continuous Deployment Pipeline (CD Pipeline)
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str, 
    min_accuracy: float = 0, 
    workers: int = 1, 
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT):
    """
    Pipeline for continuous deployment that triggers deployment only if model accuracy meets minimum threshold
    """
    # Setting up or create the experiment
    experiment_name = "continuous_deployment_pipeline"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
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

# Inference Pipeline
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(
    pipeline_name : str,
    pipeline_step_name : str
):
    data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name = pipeline_name,
        pipeline_step_name = pipeline_step_name,
        running = False,
    )
    prediction = predictor(service=service, data=data)
    return prediction
