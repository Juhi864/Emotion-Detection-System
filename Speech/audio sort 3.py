import os

# Set the path to your folder containing .wav files
folder_path = "dataset/surprised"  # Change this to your actual folder

# List all .wav files
wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]

# Sort to ensure consistent order
wav_files.sort()

# Rename files
for i, filename in enumerate(wav_files, start=1):
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, f"file{i}.wav")
    os.rename(old_path, new_path)

print(f"✅ Renamed {len(wav_files)} files successfully!")
