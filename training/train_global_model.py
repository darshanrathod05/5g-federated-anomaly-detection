import numpy as np
from tensorflow.keras.models import load_model

CLIENT1_MODEL = "models/local_model_1.h5"
CLIENT2_MODEL = "models/local_model_2.h5"
GLOBAL_MODEL_OUTPUT = "models/global_model.h5"

# =========================
# LOAD LOCAL CLIENT MODELS
# =========================
model1 = load_model(CLIENT1_MODEL)
model2 = load_model(CLIENT2_MODEL)

print("Client 1 model loaded")
print("Client 2 model loaded")

# =========================
# GET WEIGHTS
# =========================
weights1 = model1.get_weights()
weights2 = model2.get_weights()

# =========================
# FEDERATED AVERAGING
# =========================
global_weights = []

for w1, w2 in zip(weights1, weights2):
    avg_weight = np.mean([w1, w2], axis=0)
    global_weights.append(avg_weight)

# =========================
# CREATE GLOBAL MODEL
# =========================
global_model = load_model(CLIENT1_MODEL)

# Set averaged weights
global_model.set_weights(global_weights)

# Save global model
global_model.save(GLOBAL_MODEL_OUTPUT)

print("\nGlobal federated model saved at:", GLOBAL_MODEL_OUTPUT)
print("Federated averaging completed successfully")