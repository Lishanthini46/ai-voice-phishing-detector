import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pickle
import tempfile

# ======================================
# LOAD TRAINED MODEL
# ======================================
model = pickle.load(open("voice_phishing_model.pkl", "rb"))

# Number of MFCC used during training
N_MFCC = 13

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(page_title="AI Voice Phishing Detector", layout="centered")

st.title("🎙️ AI Voice Phishing Detector")
st.write("Upload an audio file to detect whether it is **Normal** or **Phishing**.")

# ======================================
# FILE UPLOAD
# ======================================
uploaded_file = st.file_uploader("Upload WAV audio file", type=["wav"])

# ======================================
# FEATURE EXTRACTION
# ======================================
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=2048,
        hop_length=512
    )

    mfcc_mean = np.mean(mfcc, axis=1)

    return y, sr, mfcc, mfcc_mean


# ======================================
# MAIN LOGIC
# ======================================
if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(uploaded_file)

    # Extract audio features
    y, sr, mfcc, mfcc_mean = extract_features(audio_path)

    # ======================================
    # AUDIO WAVEFORM
    # ======================================
    st.subheader("🔊 Audio Waveform")

    fig1, ax1 = plt.subplots(figsize=(12, 3))
    ax1.plot(y, color="black")
    ax1.set_title("Waveform")
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Amplitude")
    st.pyplot(fig1, use_container_width=True)

    # ======================================
    # FREQUENCY SPECTRUM
    # ======================================
    st.subheader("📊 Frequency Spectrum")

    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    D_db = librosa.amplitude_to_db(D, ref=np.max)

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.pcolormesh(
        D_db,
        shading="nearest",
        cmap="gray"
    )
    ax2.set_title("Frequency Spectrum (dB)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Frequency")

    st.pyplot(fig2, use_container_width=True)

    # ======================================
    # MFCC (CLEAR)
    # ======================================
    st.subheader("🎵 MFCC (Mel Frequency Cepstral Coefficients)")

    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.pcolormesh(
        mfcc,
        shading="nearest",
        cmap="gray"
    )
    ax3.set_title("MFCC Features")
    ax3.set_xlabel("Time Frames")
    ax3.set_ylabel("MFCC Index")

    st.pyplot(fig3, use_container_width=True)

    # ======================================
    # PREDICTION
    # ======================================
    if mfcc_mean.shape[0] != model.n_features_in_:
        st.error(
            f"Feature mismatch! Model expects {model.n_features_in_}, "
            f"but got {mfcc_mean.shape[0]}"
        )
        st.stop()

    prediction = model.predict([mfcc_mean])
    confidence = model.predict_proba([mfcc_mean])

    st.subheader("🔍 Detection Result")

    label = "🚨 PHISHING CALL" if prediction[0] == 1 else "✅ NORMAL CALL"
    confidence_score = np.max(confidence) * 100

    if prediction[0] == 1:
        st.error(label)
    else:
        st.success(label)

    st.info(f"📊 Confidence: {confidence_score:.2f}%")




        












