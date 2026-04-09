import shutil
import os
import re

GLOBAL_MODEL = "models/global_model.h5"
MODELS_FOLDER = "models"

# Detect local client models automatically
client_files = [
    file for file in os.listdir(MODELS_FOLDER)
    if re.match(r"local_model_\d+\.h5", file)
]

print(f"Detected {len(client_files)} clients")

# Overwrite local models directly
for client_file in client_files:
    client_path = os.path.join(MODELS_FOLDER, client_file)

    shutil.copy(GLOBAL_MODEL, client_path)

    print(f"Updated {client_file} with global model")

print("\nAll local client models updated successfully")