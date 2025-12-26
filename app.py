import streamlit as st
import numpy as np
import tempfile
import pickle
import librosa
import librosa.display
import matplotlib.pyplot as plt

# -----------------------------
# Load trained model
# -----------------------------
model = pickle.load(open("voice_phishing_model.pkl", "rb"))

# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, duration=5)

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
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Voice Phishing Detector", layout="centered")

st.title("🎙️ AI Voice Phishing Detector")
st.write("Upload a voice file to detect **Phishing or Safe** audio.")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:

    # Save file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    # Play audio
    st.audio(uploaded_file)

    # Extract features
    y, sr, mfcc, mfcc_mean = extract_features(audio_path)

    # ============================
    # MFCC GRAPH
    # ============================
    st.subheader("🎵 MFCC Feature Graph")

    mfcc_db = librosa.power_to_db(mfcc, ref=np.max)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    img1 = librosa.display.specshow(
        mfcc_db,
        x_axis="time",
        sr=sr,
        ax=ax1,
        cmap="coolwarm"
    )

    ax1.set_title("MFCC (Mel Frequency Cepstral Coefficients)", fontsize=12)
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("MFCC Coefficients")

    cbar1 = fig1.colorbar(img1, ax=ax1)
    cbar1.set_label("Amplitude (dB)")

    st.pyplot(fig1)

    # ============================
    # FREQUENCY GRAPH
    # ============================
    st.subheader("📊 Frequency Spectrum")

    D = np.abs(librosa.stft(y))

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    img2 = librosa.display.specshow(
        librosa.amplitude_to_db(D, ref=np.max),
        sr=sr,
        x_axis="time",
        y_axis="log",
        cmap="magma",
        ax=ax2
    )

    ax2.set_title("Frequency Spectrum (Log Scale)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Frequency (Hz)")

    cbar2 = fig2.colorbar(img2, ax=ax2)
    cbar2.set_label("Intensity (dB)")

    st.pyplot(fig2)

    # ============================
    # Prediction
    # ============================
    prediction = model.predict([mfcc_mean])

    st.subheader("🔍 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Phishing Voice Detected")
    else:
        st.success("✅ Safe Voice")

        


