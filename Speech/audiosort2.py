import os
import pandas as pd

# Path to dataset
DATA_PATH = "dataset"  # Change this if needed

# Check if the dataset exists
if not os.path.exists(DATA_PATH):
    raise ValueError(f"❌ Path does not exist: {DATA_PATH}")

# List of emotions found
data_records = []

# Scan dataset and count files
for label in sorted(os.listdir(DATA_PATH)):  # Sort for consistency
    label_path = os.path.join(DATA_PATH, label)
    if os.path.isdir(label_path):  # Only process folders
        files = [f for f in os.listdir(label_path) if f.endswith(".wav")]
        print(f"📂 Found {len(files)} files in '{label}' folder")
        
        # Save file names and labels
        for file in files:
            data_records.append([file, label])

# Convert to DataFrame
df = pd.DataFrame(data_records, columns=["Filename", "Emotion"])
csv_path = "emotion_labels.csv"
df.to_csv(csv_path, index=False)

print(f"✅ Dataset labeling complete. Labels saved to: {csv_path}")
