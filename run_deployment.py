from typing import cast

import click
from rich import print

from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)

from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService


DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help=(
        "Choose to only deploy (`deploy`), only predict (`predict`), "
        "or do both (`deploy_and_predict`)."
    ),
)
@click.option(
    "--min-accuracy",
    default=0.92,
    type=float,
    help="Minimum accuracy required to deploy the model.",
)
def main(config: str, min_accuracy: float):
    """Run the MLflow deployment & inference pipelines."""

    # Get active MLflow model deployer from the stack
    mlflow_model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    deploy = config in {DEPLOY, DEPLOY_AND_PREDICT}
    predict = config in {PREDICT, DEPLOY_AND_PREDICT}

    if deploy:
        print("[bold green]Running deployment pipeline...[/bold green]")
        continuous_deployment_pipeline(
            min_accuracy=min_accuracy,
            workers=3,
            timeout=60,
        )

    if predict:
        print("[bold green]Running inference pipeline...[/bold green]")
        inference_pipeline(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
        )

    print(
        "\nYou can inspect your MLflow runs using:\n"
        f"[italic green]mlflow ui --backend-store-uri '{get_tracking_uri()}'[/italic green]\n"
    )

    # Fetch running MLflow deployment services
    existing_services = mlflow_model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])

        if service.is_running:
            print(
                "\n[bold green]MLflow prediction service is running![/bold green]\n"
                f"Prediction URL:\n  {service.prediction_url}\n\n"
                "To stop the service, run:\n"
                f"[italic green]zenml model-deployer models delete {service.uuid}[/italic green]\n"
            )

        elif service.is_failed:
            print(
                "\n[bold red]MLflow prediction service failed![/bold red]\n"
                f"State: {service.status.state.value}\n"
                f"Error: {service.status.last_error}\n"
            )
    else:
        print(
            "\n[bold yellow]No MLflow prediction service found.[/bold yellow]\n"
            "Run the deployment pipeline first using:\n"
            "[italic green]python run_deployment.py --config deploy[/italic green]\n"
        )


if __name__ == "__main__":
    main()

#done for the day 