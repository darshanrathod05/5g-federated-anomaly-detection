import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import load_model

MODEL_PATH = "models/global_model.h5"
SCALER_PATH = "models/client1_scaler.pkl"
TEST_FILE = "data/processed_dataset.csv"

# Load data
df = pd.read_csv(TEST_FILE)

df["log_type"] = df["log_type"].astype("category").cat.codes

X = df.drop("label", axis=1)
y = df["label"]

# Load scaler
scaler = joblib.load(SCALER_PATH)
X_scaled = scaler.transform(X)

# Load global model
model = load_model(MODEL_PATH)

# Predict
y_pred = (model.predict(X_scaled) > 0.5).astype(int).flatten()

# Metrics
accuracy = accuracy_score(y, y_pred)

print("Global Model Accuracy:", accuracy)
print(classification_report(y, y_pred))