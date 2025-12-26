import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import pickle

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="AI Voice Phishing Detector",
    layout="centered"
)

st.title("🎙️ AI Voice Phishing Detector")

# -------------------------------
# Load model
# -------------------------------
model = pickle.load(open("voice_phishing_model.pkl", "rb"))

# -------------------------------
# Audio Feature Extraction
# -------------------------------
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)

    # Normalize audio
    y = librosa.util.normalize(y)

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    return y, sr, mfcc, mfcc_mean


# -------------------------------
# Upload Audio
# -------------------------------
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(uploaded_file)

    # Extract features
    y, sr, mfcc, mfcc_mean = extract_features(audio_path)

    # ===============================
    # 🎵 WAVEFORM
    # ===============================
    st.subheader("🎵 Audio Waveform")

    fig1, ax1 = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax1, color="blue")
    ax1.set_title("Waveform")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    st.pyplot(fig1)

    # ===============================
    # 📊 FREQUENCY SPECTRUM (CLEAR)
    # ===============================
    st.subheader("📊 Frequency Spectrum")

    D = np.abs(librosa.stft(y, n_fft=2048))
    D_db = librosa.amplitude_to_db(D, ref=np.max)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    img2 = librosa.display.specshow(
        D_db,
        sr=sr,
        x_axis="time",
        y_axis="log",
        cmap="magma",
        ax=ax2
    )
    ax2.set_title("Frequency Spectrum (Log Scale)")
    fig2.colorbar(img2, ax=ax2, format="%+2.0f dB")
    st.pyplot(fig2)

    # ===============================
    # 🎼 MFCC (CLEAR & SHARP)
    # ===============================
    st.subheader("🎼 MFCC (Mel Frequency Cepstral Coefficients)")

    mfcc_db = librosa.power_to_db(mfcc, ref=np.max)

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    img3 = librosa.display.specshow(
        mfcc_db,
        x_axis="time",
        cmap="viridis",
        ax=ax3
    )
    ax3.set_title("MFCC Features")
    ax3.set_ylabel("MFCC Coefficients")
    fig3.colorbar(img3, ax=ax3)
    st.pyplot(fig3)

    # ===============================
    # 🔍 Prediction
    # ===============================
    st.subheader("🔍 Prediction Result")

    mfcc_mean = mfcc_mean.reshape(1, -1)

    try:
        prediction = model.predict(mfcc_mean)

        if prediction[0] == 1:
            st.error("⚠️ Phishing Voice Detected")
        else:
            st.success("✅ Safe / Normal Voice")

    except Exception as e:
        st.error("Prediction error: Model feature mismatch")
        st.text(str(e))





        













