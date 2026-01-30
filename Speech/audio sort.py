import os
import shutil

source_folder = "AudioWAV"  # ← change this
target_folder = "dataset"

# Emotion abbreviation to label mapping
emotion_map = {
    'ANG': 'angry',
    'DIS': 'disgust',
    'FEA': 'fearful',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad'
}

# Create emotion folders
for emotion in set(emotion_map.values()):
    os.makedirs(os.path.join(target_folder, emotion), exist_ok=True)

# Sort files
for filename in os.listdir(source_folder):
    if filename.endswith(".wav"):
        try:
            parts = filename.split("_")
            code = parts[2]  # e.g., ANG
            emotion_label = emotion_map.get(code)
            if emotion_label:
                src = os.path.join(source_folder, filename)
                dst = os.path.join(target_folder, emotion_label, filename)
                shutil.copy2(src, dst)
            else:
                print(f"Unknown emotion code in {filename}")
        except Exception as e:
            print(f"Skipping {filename}: {e}")

print("✅ Dataset organized by emotion!")

