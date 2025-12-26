

import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pickle
import tempfile

# ------------------------------------
# Load trained model
# ------------------------------------
model = pickle.load(open("voice_phishing_model.pkl", "rb"))

# ------------------------------------
# Streamlit Page Settings
# ------------------------------------
st.set_page_config(page_title="AI Voice Phishing Detector", layout="centered")

st.title("🎙️ AI Voice Phishing Detector")
st.write("Upload an audio file to check whether it is **Phishing** or **Normal**.")

# ------------------------------------
# File Upload
# ------------------------------------
uploaded_file = st.file_uploader("Upload WAV audio file", type=["wav"])

# ------------------------------------
# Feature Extraction
# ------------------------------------
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    # MFCC (13 features)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    return y, sr, mfcc, mfcc_mean


# ------------------------------------
# Main Logic
# ------------------------------------
if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(uploaded_file)

    # Extract features
    y, sr, mfcc, mfcc_mean = extract_features(audio_path)

    # ==============================
    # WAVEFORM (CLEAR)
    # ==============================
    st.subheader("🔊 Audio Waveform")

    fig1, ax1 = plt.subplots(figsize=(10, 3))
    ax1.plot(y, color="black")
    ax1.set_title("Waveform")
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Amplitude")
    st.pyplot(fig1)

    # ==============================
    # FREQUENCY SPECTRUM (CLEAR)
    # ==============================
    st.subheader("📊 Frequency Spectrum")

    D = np.abs(librosa.stft(y))
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    img2 = ax2.imshow(
        librosa.amplitude_to_db(D, ref=np.max),
        origin="lower",
        aspect="auto",
        cmap="gray"
    )
    ax2.set_title("Frequency Spectrum")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)

    # ==============================
    # MFCC (VERY CLEAR)
    # ==============================
    st.subheader("🎵 MFCC (Clear View)")

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    img3 = ax3.imshow(
        mfcc,
        aspect="auto",
        origin="lower",
        cmap="gray"
    )
    ax3.set_title("MFCC Coefficients")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("MFCC Index")
    st.pyplot(fig3)

    # ==============================
    # Prediction
    # ==============================
    prediction = model.predict([mfcc_mean])
    confidence = model.predict_proba([mfcc_mean])

    st.subheader("🔍 Detection Result")

    label = "PHISHING CALL 🚨" if prediction[0] == 1 else "NORMAL CALL ✅"
    confidence_score = np.max(confidence) * 100

    if prediction[0] == 1:
        st.error(label)
    else:
        st.success(label)

    st.info(f"📊 Confidence: {confidence_score:.2f}%")

        









