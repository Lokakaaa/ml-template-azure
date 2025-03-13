from azureml.core import Workspace, Model
import os
from datetime import datetime

# Load workspace from config.json or Azure credentials
ws = Workspace.from_config()

# Define model name and path
model_name = "Hydra_EU"
model_version_tag = datetime.now().strftime('%Y%m%d')
model_path = f"outputs/{model_name}_{model_version_tag}.pkl"

# Ensure the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Register the model
model = Model.register(
    workspace=ws,
    model_name=model_name,  # Keeps model versioning under one name
    model_path=model_path,
    description=f"Model retrained for {model_version_tag}",
    tags={"retrain_date": model_version_tag}
)

print(f"✅ Registered Model: {model.name}, Version: {model.version}")
