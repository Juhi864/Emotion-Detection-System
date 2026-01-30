
import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import pickle
import os
import tempfile

st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")

st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a `.wav` audio file and detect the emotion from speech!")

@st.cache_resource
def load_emotion_model():
    return load_model("emotion_model.h5")

@st.cache_resource
def load_label_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)

def extract_features(file_path, max_pad_len=174):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs

model = load_emotion_model()
encoder = load_label_encoder()

uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file, format="audio/wav")

    features = extract_features(tmp_path)
    features = features.reshape(1, 40, 174, 1)
    prediction = model.predict(features)
    predicted_emotion = encoder.inverse_transform([np.argmax(prediction)])[0]

    st.success(f"🧠 Detected Emotion: **{predicted_emotion}**")
