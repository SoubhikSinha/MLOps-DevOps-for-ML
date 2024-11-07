'''
Model Deployment (Local Machine)
'''

# Importing necessary libraries
import click
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from pipelines.deployment_pipeline import continuous_deployment_pipeline, inference_pipeline
from typing import cast

# Constants for configuration choices
DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Choose to run deployment pipeline ('deploy') or prediction ('predict'). Default is both ('deploy_and_predict')."
)
@click.option(
    "--min-accuracy",
    default=0.0,
    type=float,
    help="Minimum accuracy required to deploy the model"
)

def run_deployment(config: str, min_accuracy: float):
    # Initialize model deployer component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    # Run the deployment pipeline if 'deploy' is chosen
    if deploy:
        continuous_deployment_pipeline(
            data_path="D:/GitHub_Repos/MLOps-DevOps-for-ML/data/olist_customers_dataset.csv",
            min_accuracy=min_accuracy,
            workers=3,
            timeout=60,
        )

    # Run the inference pipeline if 'predict' is chosen
    if predict:
        inference_pipeline(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
        )

    # Output MLflow UI command
    print(
        "You can run:\n"
        f"[italic green]    mlflow ui --backend-store-uri '{get_tracking_uri()}'[/italic green]\n"
        "...to inspect your experiment runs within the MLflow UI.\n"
        "You can find your runs tracked within the 'mlflow_example_pipeline' experiment.\n"
    )

    # Find any existing model server
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model"
    )

    # Check service status and output information
    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(
                f"The MLflow prediction server is running at:\n   {service.prediction_url}\n"
                f"[italic green]`zenml model-deployer models delete {str(service.uuid)}`[/italic green]"
            )
        elif service.is_failed:
            print(
                f"The MLflow prediction server is in a failed state:\n"
                f" Last state: '{service.status.state.value}'\n"
                f" Last error: '{service.status.last_error}'"
            )
    else:
        print(
            "No MLflow prediction server is currently running. Run the deployment pipeline with the `--deploy` option."
        )

if __name__ == "__main__":
    run_deployment()