import streamlit as st
import numpy as np
import tempfile
import pickle
import matplotlib.pyplot as plt
import librosa

# -----------------------------
# Load trained model
# -----------------------------
model = pickle.load(open("voice_phishing_model.pkl", "rb"))

# -----------------------------
# Feature extraction
# -----------------------------
def extract_mfcc(audio_path):
    y, sr = librosa.load(audio_path, duration=5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc, mfcc_mean


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Voice Phishing Detector", layout="centered")

st.title("🎙️ AI Voice Phishing Detector")
st.write("Upload an audio file to detect whether it is **Phishing** or **Safe**.")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

# -----------------------------
# Process Audio
# -----------------------------
if uploaded_file is not None:

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    # Play audio
    st.audio(uploaded_file)

    # Extract features
    mfcc, mfcc_mean = extract_mfcc(audio_path)

    # -----------------------------
    # MFCC Visualization (SAFE)
    # -----------------------------
    st.subheader("🎵 MFCC Feature Visualization")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(mfcc, aspect="auto", origin="lower")
    ax.set_title("MFCC Coefficients")
    ax.set_xlabel("Time")
    ax.set_ylabel("MFCC Coefficients")

    st.pyplot(fig)

    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict([mfcc_mean])

    st.subheader("🔍 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Phishing Voice Detected")
    else:
        st.success("✅ Safe Voice")

    

   
