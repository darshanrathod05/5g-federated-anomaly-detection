import pandas as pd
import re

INPUT_FILE = "data/our_Logs_Cleaned.csv"
OUTPUT_FILE = "data/processed_dataset.csv"

# Load cleaned dataset
df = pd.read_csv(INPUT_FILE)

print("Columns:", df.columns)

LOG_COLUMN = "log"


def extract_features(log):
    log = str(log)

    features = {}

    # RX
    rx_match = re.search(r'rx\s+(\d+)', log)
    features["rx"] = int(rx_match.group(1)) if rx_match else 0

    # TX
    tx_match = re.search(r'tx\s+(\d+)', log)
    features["tx"] = int(tx_match.group(1)) if tx_match else 0

    # PPS
    pps_match = re.search(r'pps\s+(\d+)', log)
    features["pps"] = int(pps_match.group(1)) if pps_match else 0

    # KBPS
    kbps_match = re.search(r'kbps\s+(\d+)', log)
    features["kbps"] = int(kbps_match.group(1)) if kbps_match else 0

    # Frame and Slot from "Frame.Slot 512.0"
    frame_slot_match = re.search(r'Frame\.Slot\s+(\d+)\.(\d+)', log)

    if frame_slot_match:
        features["frame"] = int(frame_slot_match.group(1))
        features["slot"] = int(frame_slot_match.group(2))
    else:
        features["frame"] = 0
        features["slot"] = 0

    # BLER
    bler_match = re.search(r'BLER\s+([0-9.]+)', log)
    features["BLER"] = float(bler_match.group(1)) if bler_match else 0

    # RSRP
    rsrp_match = re.search(r'RSRP\s+(-?\d+)', log)
    features["RSRP"] = int(rsrp_match.group(1)) if rsrp_match else 0

    # PRACH
    prach_match = re.search(r'PRACH\s+(\d+)', log)
    features["PRACH"] = int(prach_match.group(1)) if prach_match else 0

    # PUSCH
    pusch_match = re.search(r'PUSCH\s+(\d+)', log)
    features["PUSCH"] = int(pusch_match.group(1)) if pusch_match else 0

    # Warning/Error flag
    features["warning_flag"] = 1 if (
        "warning" in log.lower() or "error" in log.lower()
    ) else 0

    # Generic log type
    type_match = re.search(r'\[([^\]]+)\]', log)
    features["log_type"] = type_match.group(1).strip() if type_match else "UNKNOWN"

    # RX-TX imbalance ratio
    rx = features["rx"]
    tx = features["tx"]

    if max(rx, tx) != 0:
        features["rx_tx_ratio"] = abs(rx - tx) / max(rx, tx)
    else:
        features["rx_tx_ratio"] = 0

    return features


# Apply extraction
features_df = df[LOG_COLUMN].apply(extract_features).apply(pd.Series)


def create_label(row):
    # RX-TX imbalance
    if row["rx_tx_ratio"] > 0.4:
        return 1

    # PPS anomaly
    if row["pps"] != 0 and (row["pps"] < 5000 or row["pps"] > 100000):
        return 1

    # KBPS anomaly
    if row["kbps"] != 0 and (row["kbps"] < 10000 or row["kbps"] > 100000):
        return 1

    # Frame anomaly
    if row["frame"] < 0:
        return 1

    # Slot anomaly
    if row["slot"] != 0 and (row["slot"] < 0 or row["slot"] > 19):
        return 1

    # BLER anomaly
    if row["BLER"] != 0 and row["BLER"] > 0.2:
        return 1

    # RSRP anomaly
    if row["RSRP"] != 0 and row["RSRP"] < -105:
        return 1

    # PRACH anomaly
    if row["PRACH"] != 0 and row["PRACH"] > 5000000:
        return 1

    # PUSCH anomaly
    if row["PUSCH"] != 0 and row["PUSCH"] > 100000000:
        return 1

    # Warning/Error anomaly
    if row["warning_flag"] == 1:
        return 1

    return 0


# Create labels
features_df["label"] = features_df.apply(create_label, axis=1)

# Save dataset
features_df.to_csv(OUTPUT_FILE, index=False)

print("Processed dataset saved successfully")
print(features_df.head())
print(features_df["label"].value_counts())