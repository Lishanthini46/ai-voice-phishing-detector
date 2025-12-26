import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pickle
import tempfile

# -----------------------------
# Load trained model
# -----------------------------
model = pickle.load(open("voice_phishing_model.pkl", "rb"))

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI Voice Phishing Detector", layout="centered")

st.title("🎙️ AI Voice Phishing Detector")
st.write("Upload an audio file to detect whether it is **Normal** or **Phishing**.")

# -----------------------------
# Upload Audio
# -----------------------------
uploaded_file = st.file_uploader("Upload WAV audio file", type=["wav"])

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40,
        n_fft=2048,
        hop_length=512
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    return y, sr, mfcc, mfcc_mean


# -----------------------------
# MAIN PROCESS
# -----------------------------
if uploaded_file is not None:

    # Save audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(uploaded_file)

    # Extract features
    y, sr, mfcc, mfcc_mean = extract_features(audio_path)

    # =====================================================
    # 1️⃣ AUDIO WAVEFORM
    # =====================================================
    st.subheader("🔊 Audio Waveform")

    fig1, ax1 = plt.subplots(figsize=(12, 3), dpi=120)
    ax1.plot(y, color="black")
    ax1.set_title("Waveform")
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Amplitude")
    st.pyplot(fig1, use_container_width=True)

    # =====================================================
    # 2️⃣ FREQUENCY SPECTRUM
    # =====================================================
    st.subheader("📊 Frequency Spectrum")

    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    D_db = librosa.amplitude_to_db(D, ref=np.max)

    fig2, ax2 = plt.subplots(figsize=(12, 4), dpi=120)
    img2 = ax2.imshow(
        D_db,
        origin="lower",
        aspect="auto",
        cmap="gray",
        interpolation="nearest"
    )

    ax2.set_title("Frequency Spectrum (dB)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Frequency")

    st.pyplot(fig2, use_container_width=True)

    # =====================================================
    # 3️⃣ MFCC (CLEAR & SHARP)
    # =====================================================
    st.subheader("🎵 MFCC (Mel Frequency Cepstral Coefficients)")

    fig3, ax3 = plt.subplots(figsize=(12, 4), dpi=120)
    img3 = ax3.imshow(
        mfcc,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        interpolation="nearest"
    )

    ax3.set_title("MFCC Representation")
    ax3.set_xlabel("Time Frames")
    ax3.set_ylabel("MFCC Coefficients")

    st.pyplot(fig3, use_container_width=True)

    # =====================================================
    # 4️⃣ PREDICTION
    # =====================================================
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


        










