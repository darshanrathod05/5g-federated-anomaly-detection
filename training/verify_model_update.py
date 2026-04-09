import numpy as np
from tensorflow.keras.models import load_model

global_model = load_model("models/global_model.h5")
local_model = load_model("models/local_model_1.h5")

global_weights = global_model.get_weights()
local_weights = local_model.get_weights()

same = True

for gw, lw in zip(global_weights, local_weights):
    if not np.array_equal(gw, lw):
        same = False
        break

if same:
    print("SUCCESS: Local model updated from global model")
else:
    print("ERROR: Local model not updated")