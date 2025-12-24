import streamlit as st
import librosa
import numpy as np
import pickle

st.set_page_config(page_title="AI Voice Phishing Detector")

st.title("📞 AI Voice Phishing Detector")
st.write("Upload a call recording (.wav) to detect phishing.")

# Load trained model
with open("voice_phishing_model.pkl", "rb") as f:
    model = pickle.load(f)

# Feature extraction
def extract_features(file):
    audio, sr = librosa.load(file, duration=5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    features = extract_features(uploaded_file)
    prediction = model.predict([features])

    if prediction[0] == 1:
        st.error("⚠️ SCAM / PHISHING CALL DETECTED")
    else:
        st.success("✅ NORMAL / SAFE CALL")
