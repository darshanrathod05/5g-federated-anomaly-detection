import pandas as pd

INPUT_FILE  = "data/processed_dataset.csv"

CLIENT1_OUTPUT = "data/client1_dataset.csv"
CLIENT2_OUTPUT = "data/client2_dataset.csv"

# Load dataset
df = pd.read_csv(INPUT_FILE)
print("Total rows:", len(df))

# Shuffle before splitting
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split 50/50
split_index = len(df) // 2
client1_df = df.iloc[:split_index]
client2_df = df.iloc[split_index:]

# Save
client1_df.to_csv(CLIENT1_OUTPUT, index=False)
client2_df.to_csv(CLIENT2_OUTPUT, index=False)

print("Client 1 rows:", len(client1_df), "->", CLIENT1_OUTPUT)
print("Client 2 rows:", len(client2_df), "->", CLIENT2_OUTPUT)