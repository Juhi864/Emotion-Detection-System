

import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle

def extract_features(file_path, max_pad_len=174):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs

model = load_model('emotion_model.h5')

with open('label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

def predict_emotion(file_path):
    mfccs = extract_features(file_path)
    mfccs = mfccs.reshape(1, 40, 174, 1)
    prediction = model.predict(mfccs)
    emotion = encoder.inverse_transform([np.argmax(prediction)])
    return emotion[0]




# Example usage
# print("Predicted Emotion:", predict_emotion("test_audio.wav"))
