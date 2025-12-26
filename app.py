import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pickle
import tempfile

# -----------------------------
# Load model
# -----------------------------
model = pickle.load(open("voice_phishing_model.pkl", "rb"))

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="AI Voice Phishing Detector", layout="centered")

st.title("🎙️ AI Voice Phishing Detector")
st.write("Upload an audio file to detect **Phishing or Normal voice**")

# -----------------------------
# Upload audio
# -----------------------------
uploaded_file = st.file_uploader("Upload WAV audio file", type=["wav"])

# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    # IMPORTANT: same as training
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    return y, sr, mfcc, mfcc_mean


# -----------------------------
# MAIN
# -----------------------------
if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(uploaded_file)

    y, sr, mfcc, mfcc_mean = extract_features(audio_path)

    # ===============================
    # WAVEFORM
    # ===============================
    st.subheader("🔊 Audio Waveform")

    fig1, ax1 = plt.subplots(figsize=(12, 3))
    ax1.plot(y, color="black")
    ax1.set_title("Waveform")
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Amplitude")
    st.pyplot(fig1, use_container_width=True)

    # ===============================
    # FREQUENCY SPECTRUM (CLEAR)
    # ===============================
    st.subheader("📊 Frequency Spectrum")

    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    D_db = librosa.amplitude_to_db(D, ref=np.max)

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    img2 = ax2.pcolormesh(
        D_db,
        shading="nearest",
        cmap="gray"
    )

    ax2.set_title("Frequency Spectrum (dB)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Frequency")

    st.pyplot(fig2, use_container_width=True)

    # ===============================
    # MFCC (CLEAR & SHARP)
    # ===============================
    st.subheader("🎵 MFCC (13 Coefficients)")

    fig3, ax3 = plt.subplots(figsize=(12, 4))
    img3 = ax3.pcolormesh(
        mfcc,
        shading="nearest",
        cmap="gray"
    )

    ax3.set_title("MFCC Features")
    ax3.set_xlabel("Time Frames")
    ax3.set_ylabel("MFCC Index")

    st.pyplot(fig3, use_container_width=True)

    # ===============================
    # PREDICTION
    # ===============================
    prediction = model.predict([mfcc_mean])
    confidence = model.predict_proba([mfcc_mean])

    st.subheader("🔍 Detection Result")

    label = "🚨 PHISHING CALL" if prediction[0] == 1 else "✅ NORMAL CALL"
    conf = np.max(confidence) * 100

    if prediction[0] == 1:
        st.error(label)
    else:
        st.success(label)

    st.info(f"📊 Confidence: {conf:.2f}%")



        











