import subprocess
import os
import re

ROUNDS = 3
TRAINING_FOLDER = "training"

# Detect all client training scripts automatically
client_scripts = [
    file for file in os.listdir(TRAINING_FOLDER)
    if re.match(r"train_client\d+\.py", file)
]

print(f"Detected {len(client_scripts)} clients")

for round_num in range(ROUNDS):
    print(f"\n===== FEDERATED ROUND {round_num + 1} =====")

    # Run all clients
    for client_script in client_scripts:
        subprocess.run(["python", f"{TRAINING_FOLDER}/{client_script}"])

    # Global steps
    subprocess.run(["python", "training/train_global_model.py"])
    subprocess.run(["python", "training/send_global_to_clients.py"])
    subprocess.run(["python", "training/evaluate_global_model.py"])

    print(f"Round {round_num + 1} completed")

print("\nAutomatic federated learning pipeline completed")