import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "data/RAN_Logs.csv"
OUTPUT_FILE = "data/RAN_Logs_Cleaned.csv"

# ── Column mapping: raw → cleaned ─────────────────────────────────────────────
COLUMN_MAP = {
    "@log_name"  : "log_name",
    "@timestamp" : "timestamp",
    "_id"        : "id",
    "_index"     : "index",
    "container_id"   : "container_id",
    "container_name" : "container_name",
    "log"        : "log",
    "source"     : "source",
}

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE, on_bad_lines="skip")

# ── Select & rename only the required columns ─────────────────────────────────
df_cleaned = df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP)

# ── Parse & reformat timestamp  (e.g. "Sep 18, 2025 @ 17:44:20.000" → "2025-09-18 17:44:20") ──
df_cleaned["timestamp"] = (
    pd.to_datetime(df_cleaned["timestamp"], format="%b %d, %Y @ %H:%M:%S.%f", errors="coerce")
    .dt.strftime("%Y-%m-%d %H:%M:%S")
)

# ── Drop rows where log column is "(empty)" ───────────────────────────────────
df_cleaned = df_cleaned[df_cleaned["log"] != "(empty)"].reset_index(drop=True)

# ── Save ─────
df_cleaned.to_csv(OUTPUT_FILE, index=False)
print(f"Done! {len(df_cleaned)} rows written to '{OUTPUT_FILE}'")
print(df_cleaned.head())