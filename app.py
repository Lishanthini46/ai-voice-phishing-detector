import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import pickle

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="AI Voice Phishing Detector",
    layout="centered"
)

# -----------------------------
# Load ML model
# -----------------------------
model = pickle.load(open("voice_phishing_model.pkl", "rb"))

# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None, duration=5)

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512
    )

    mfcc_mean = np.mean(mfcc.T, axis=0)
    return y, sr, mfcc, mfcc_mean


# -----------------------------
# UI
# -----------------------------
st.title("🎙️ AI Voice Phishing Detector")
st.write("Upload an audio file to detect whether it is **Phishing** or **Safe**.")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(uploaded_file)

    # Extract features
    y, sr, mfcc, mfcc_mean = extract_features(audio_path)

    # ==========================
    # MFCC GRAPH
    # ==========================
    st.subheader("🎵 MFCC Feature Graph")

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mfcc,
        x_axis="time",
        sr=sr,
        cmap="coolwarm",
        ax=ax1
    )

    ax1.set_title("MFCC (Mel Frequency Cepstral Coefficients)")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("MFCC Coefficients")

    cbar = fig1.colorbar(img, ax=ax1, format="%+2.0f dB")
    cbar.set_label("Amplitude (dB)")

    st.pyplot(fig1)

    # ==========================
    # FREQUENCY SPECTRUM
    # ==========================
    st.subheader("📊 Frequency Spectrum")

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    img2 = librosa.display.specshow(
        D,
        sr=sr,
        x_axis="time",
        y_axis="log",
        cmap="magma",
        ax=ax2
    )

    ax2.set_title("Frequency Spectrum (Log Scale)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Frequency (Hz)")

    cbar2 = fig2.colorbar(img2, ax=ax2, format="%+2.0f dB")
    cbar2.set_label("Intensity (dB)")

    st.pyplot(fig2)

    # ==========================
    # Prediction
    # ==========================
    prediction = model.predict([mfcc_mean])

    st.subheader("🔍 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Phishing Voice Detected")
    else:
        st.success("✅ Safe Voice")


        



