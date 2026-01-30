import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pickle

# Path to the organized dataset
DATA_PATH = "dataset"

# Function to extract MFCC features
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print(f"Error extracting from {file_path}: {e}")
        return None

# Function to load data
def load_data(data_path):
    features, labels = [], []
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(label_path, file)
                    mfcc = extract_features(file_path)
                    if mfcc is not None:
                        features.append(mfcc)
                        labels.append(label)
    return np.array(features), np.array(labels)

# Load features and labels
X, y = load_data(DATA_PATH)

# Safety check
if X.shape[0] == 0:
    raise ValueError("❌ No data found. Check your dataset path and structure.")

# Reshape for CNN
X = X.reshape(X.shape[0], 40, 174, 1)

# Label encoding
encoder = LabelEncoder()
y_encoded = to_categorical(encoder.fit_transform(y))

# Save the label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(40, 174, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_encoded.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
print(f"❌ Loss: {loss:.4f}")


# Save model
model.save("emotion_model.h5")
print("✅ Model trained and saved as emotion_model.h5")
