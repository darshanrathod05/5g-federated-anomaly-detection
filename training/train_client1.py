import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

INPUT_FILE = "data/client1_dataset.csv"
MODEL_OUTPUT = "models/local_model_1.h5"
SCALER_OUTPUT = "models/client1_scaler.pkl"

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)
print(df.head())

# =========================
# PREPROCESSING
# =========================

# Convert categorical column
df["log_type"] = df["log_type"].astype("category").cat.codes

# Features and label
X = df.drop("label", axis=1)
y = df["label"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler (important for future prediction/global model)
joblib.dump(scaler, SCALER_OUTPUT)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# DEEP LEARNING MODEL
# =========================
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAIN MODEL
# =========================
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# =========================
# EVALUATION
# =========================
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)

print("\nClient 1 Deep Learning Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# SAVE LOCAL MODEL
# =========================
model.save(MODEL_OUTPUT)

print("\nLocal deep learning model saved at:", MODEL_OUTPUT)
print("Scaler saved at:", SCALER_OUTPUT)