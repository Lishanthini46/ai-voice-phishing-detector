import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import pickle

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(
    page_title="AI Voice Phishing Detector",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(open("voice_phishing_model.pkl", "rb"))

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20,
        n_fft=2048,
        hop_length=512
    )

    mfcc_mean = np.mean(mfcc.T, axis=0)
    return y, sr, mfcc, mfcc_mean


# -----------------------------
# UI
# -----------------------------
st.title("🎙️ AI Voice Phishing Detector")
st.write("Upload a **WAV audio file** for analysis")

uploaded_file = st.file_uploader("Upload audio", type=["wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(uploaded_file)

    y, sr, mfcc, mfcc_mean = extract_features(audio_path)

    # ===============================
    # 🔊 WAVEFORM
    # ===============================
    st.subheader("🔊 Audio Waveform")

    fig_wave, ax = plt.subplots(figsize=(12, 3), dpi=150)
    librosa.display.waveshow(y, sr=sr, ax=ax, color="blue")
    ax.set_title("Waveform")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig_wave)

    # ===============================
    # 🎵 MFCC (CLEAR)
    # ===============================
    st.subheader("🎵 MFCC Feature Map")

    mfcc_db = librosa.power_to_db(mfcc, ref=np.max)

    fig_mfcc, ax = plt.subplots(figsize=(12, 4), dpi=150)

    img1 = librosa.display.specshow(
        mfcc_db,
        x_axis="time",
        sr=sr,
        cmap="magma",
        vmin=-80,
        vmax=0,
        ax=ax
    )

    ax.set_title("MFCC (High Clarity)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("MFCC Coefficients")

    cbar1 = fig_mfcc.colorbar(img1, ax=ax)
    cbar1.set_label("Intensity (dB)")

    st.pyplot(fig_mfcc)

    # ===============================
    # 📊 FREQUENCY SPECTRUM
    # ===============================
    st.subheader("📊 Frequency Spectrum")

    D = librosa.stft(y, n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    fig_spec, ax = plt.subplots(figsize=(12, 4), dpi=150)

    img2 = librosa.display.specshow(
        S_db,
        sr=sr,
        x_axis="time",
        y_axis="log",
        cmap="inferno",
        vmin=-80,
        vmax=0,
        ax=ax
    )

    ax.set_title("Frequency Spectrum (Log Scale)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")

    cbar2 = fig_spec.colorbar(img2, ax=ax)
    cbar2.set_label("Intensity (dB)")

    st.pyplot(fig_spec)

    # ===============================
    # 🔍 Prediction
    # ===============================
    prediction = model.predict([mfcc_mean])

    st.subheader("🔍 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Phishing Voice Detected")
    else:
        st.success("✅ Safe Voice")




        






