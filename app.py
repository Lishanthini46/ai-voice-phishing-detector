import streamlit as st
import librosa
import numpy as np
import pickle
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Voice Phishing Detector",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ AI Voice Phishing Detector")
st.write("Upload a voice call recording to detect phishing or scam calls.")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# -----------------------------
# Feature Extraction Function
# -----------------------------
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    with st.spinner("🔍 Analyzing audio..."):
        features = extract_features(uploaded_file)
        features = features.reshape(1, -1)

        prediction = model.predict(features)[0]

    # -----------------------------
    # Prediction Result
    # -----------------------------
    if prediction == 1:
        st.error("🚨 Phishing / Scam Call Detected!")
    else:
        st.success("✅ This call seems to be Genuine.")

    # -----------------------------
    # MFCC Visualization
    # -----------------------------
    y, sr = librosa.load(uploaded_file, duration=30)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    fig, ax = plt.subplots()
    img = ax.imshow(mfccs, aspect='auto', origin='lower')
    plt.colorbar(img, ax=ax)
    ax.set_title("MFCC Features")
    ax.set_xlabel("Time")
    ax.set_ylabel("MFCC Coefficients")

    st.pyplot(fig)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("🔐 **Cyber Security Project – AI Voice Phishing Detection**")




