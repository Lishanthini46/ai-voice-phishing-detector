import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import pickle

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(page_title="AI Voice Phishing Detector", layout="centered")

# ---------------------------------
# Load Model
# ---------------------------------
model = pickle.load(open("voice_phishing_model.pkl", "rb"))

# ---------------------------------
# Feature Extraction
# ---------------------------------
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return y, sr, mfcc, mfcc_mean


# ---------------------------------
# UI
# ---------------------------------
st.title("🎙️ AI Voice Phishing Detector")
st.write("Upload a WAV audio file")

uploaded_file = st.file_uploader("Upload audio file", type=["wav"])

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(uploaded_file)

    y, sr, mfcc, mfcc_mean = extract_features(audio_path)

    # ==================================================
    # 🔊 WAVEFORM (CLEAR)
    # ==================================================
    st.subheader("🔊 Audio Waveform")

    fig1, ax1 = plt.subplots(figsize=(14, 3), dpi=200)
    ax1.plot(y, color="black", linewidth=0.8)
    ax1.set_title("Waveform")
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Amplitude")
    ax1.grid(alpha=0.3)

    st.pyplot(fig1)

    # ==================================================
    # 🎵 MFCC (BLACK & WHITE – SHARP)
    # ==================================================
    st.subheader("🎵 MFCC (High Clarity)")

    mfcc_db = librosa.power_to_db(mfcc, ref=np.max)

    fig2, ax2 = plt.subplots(figsize=(14, 5), dpi=200)

    img1 = ax2.imshow(
        mfcc_db,
        origin="lower",
        aspect="auto",
        cmap="gray",
        interpolation="nearest"
    )

    ax2.set_title("MFCC Coefficients")
    ax2.set_xlabel("Time Frames")
    ax2.set_ylabel("MFCC Index")

    cbar1 = fig2.colorbar(img1, ax=ax2)
    cbar1.set_label("dB")

    st.pyplot(fig2)

    # ==================================================
    # 📊 FREQUENCY SPECTRUM (CLEAR)
    # ==================================================
    st.subheader("📊 Frequency Spectrum")

    D = librosa.stft(y, n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    fig3, ax3 = plt.subplots(figsize=(14, 5), dpi=200)

    img2 = ax3.imshow(
        S_db,
        origin="lower",
        aspect="auto",
        cmap="gray",
        interpolation="nearest"
    )

    ax3.set_title("Frequency Spectrum")
    ax3.set_xlabel("Time Frames")
    ax3.set_ylabel("Frequency Bins")

    cbar2 = fig3.colorbar(img2, ax=ax3)
    cbar2.set_label("dB")

    st.pyplot(fig3)

    # ==================================================
    # 🔍 Prediction
    # ==================================================
    prediction = model.predict([mfcc_mean])

    st.subheader("🔍 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Phishing Voice Detected")
    else:
        st.success("✅ Safe Voice")


        








