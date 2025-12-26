import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import pickle

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Voice Phishing Detector",
    layout="centered"
)

# -----------------------------
# Load model
# -----------------------------
model = pickle.load(open("voice_phishing_model.pkl", "rb"))

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000, duration=5)

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
st.write("Upload a **WAV audio file** to analyze the voice.")

uploaded_file = st.file_uploader("Upload audio file", type=["wav"])

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(uploaded_file)

    # Extract features
    y, sr, mfcc, mfcc_mean = extract_features(audio_path)

    # ====================================================
    # 1️⃣ WAVEFORM
    # ====================================================
    st.subheader("🔊 Audio Waveform")

    fig_wave, ax_wave = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, alpha=0.8, ax=ax_wave)
    ax_wave.set_title("Waveform")
    ax_wave.set_xlabel("Time (seconds)")
    ax_wave.set_ylabel("Amplitude")

    st.pyplot(fig_wave)

    # ====================================================
    # 2️⃣ MFCC GRAPH (CLEAR)
    # ====================================================
    st.subheader("🎵 MFCC Feature Graph")

    fig_mfcc, ax_mfcc = plt.subplots(figsize=(10, 4))

    mfcc_db = librosa.power_to_db(mfcc, ref=np.max)

    img1 = librosa.display.specshow(
        mfcc_db,
        x_axis="time",
        sr=sr,
        cmap="plasma",
        ax=ax_mfcc
    )

    ax_mfcc.set_title("MFCC (Mel Frequency Cepstral Coefficients)")
    ax_mfcc.set_xlabel("Time (seconds)")
    ax_mfcc.set_ylabel("MFCC Coefficients")

    cbar1 = fig_mfcc.colorbar(img1, ax=ax_mfcc)
    cbar1.set_label("Intensity (dB)")

    st.pyplot(fig_mfcc)

    # ====================================================
    # 3️⃣ FREQUENCY SPECTRUM
    # ====================================================
    st.subheader("📊 Frequency Spectrum")

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    fig_spec, ax_spec = plt.subplots(figsize=(10, 4))

    img2 = librosa.display.specshow(
        D,
        sr=sr,
        x_axis="time",
        y_axis="log",
        cmap="inferno",
        ax=ax_spec
    )

    ax_spec.set_title("Frequency Spectrum (Log Scale)")
    ax_spec.set_xlabel("Time (seconds)")
    ax_spec.set_ylabel("Frequency (Hz)")

    cbar2 = fig_spec.colorbar(img2, ax=ax_spec)
    cbar2.set_label("Intensity (dB)")

    st.pyplot(fig_spec)

    # ====================================================
    # 4️⃣ PREDICTION
    # ====================================================
    prediction = model.predict([mfcc_mean])

    st.subheader("🔍 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Phishing Voice Detected")
    else:
        st.success("✅ Safe Voice")



        





