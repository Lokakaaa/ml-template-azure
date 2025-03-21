from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment, ManagedOnlineEndpoint
from azure.identity import DefaultAzureCredential
import datetime
import json
import mlflow
import requests
import pandas as pd
from mlflow.deployments import get_deploy_client
from mlflow.tracking import MlflowClient

deployment_name = "hydra-mlops-endpoint" + datetime.datetime.now().strftime("%m%d%H%M%f")
endpoint_name="hydra-mlops-endpoint"
subscription_id = "6ebf1b40-0cf8-41ef-b8b9-19fbd5aa1ace"
resource_group = "hydra-eu-rg"
workspace_name = "hydra-eu-ws"

# Initialize the ML client
ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace_name)

# initialize mlflow client
mlflow_client = MlflowClient()
# Configure the deployment client
deployment_client = get_deploy_client(mlflow.get_tracking_uri())


# Retrieve the registered model (using latest or a specific version)
model_name = "my-registered-model"
latest_model = mlflow_client.models.get(name=model_name, label="latest")

# Creating a unique endpoint name with current datetime to avoid conflicts
endpoint_name = "sklearn-diabetes-" + datetime.datetime.now().strftime("%m%d%H%M%f")

# Writen endpoint configuration using a configuration file 
endpoint_config = {
    "auth_mode": "key",
    "identity": {
        "type": "system_assigned"
    }
}

# Write the endpoint configurations
endpoint_config_path = "endpoint_config.json"
with open(endpoint_config_path, "w") as outfile:
    outfile.write(json.dumps(endpoint_config))

# create the endpoint
endpoint = deployment_client.create_endpoint(
    name=endpoint_name,
    config={"endpoint-config-file": endpoint_config_path},
)

# Deploy the model
deploy_config = {
    "instance_type": "Standard_F4s_v2",
    "instance_count": 1,
}

# Write the deployment configurations
deployment_config_path = "deployment_config.json"
with open(deployment_config_path, "w") as outfile:
    outfile.write(json.dumps(deploy_config))

# # Deploy the model
# deployment = ManagedOnlineDeployment(
#     name=deployment_name,
#     endpoint_name=endpoint_name,
#     model=latest_model,
#     instance_type="Standard_DS3_v2",
#     instance_count=1
# )

blue_deployment = deployment_client.create_deployment(
    name=blue_deployment_name,
    endpoint=endpoint_name,
    model_uri=f"models:/{model_name}/{version}",
    config={"deploy-config-file": deployment_config_path},
)

traffic_config = {"traffic": {blue_deployment_name: 100}}
traffic_config_path = "traffic_config.json"
with open(traffic_config_path, "w") as outfile:
    outfile.write(json.dumps(traffic_config))


deployment_client.update_endpoint(
    endpoint=endpoint_name,
    config={"endpoint-config-file": traffic_config_path},
)


# Optionally stream logs if required
# For streaming logs, you would use:
# ml_client.online_deployments.get_logs(deployment.name)
