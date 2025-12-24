import streamlit as st
import librosa
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="AI Voice Phishing Detector",
    page_icon="📞",
    layout="centered"
)

# Load model
with open("voice_phishing_model.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.markdown("<h1 style='text-align: center;'>📞 AI Voice Phishing Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Audio-based scam call detection using AI</p>", unsafe_allow_html=True)

st.divider()

# Upload section
st.subheader("🎧 Upload Call Recording (.wav)")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

# Feature extraction
def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfcc, np.mean(mfcc.T, axis=0)

if uploaded_file is not None:
    # Load audio
    audio, sr = librosa.load(uploaded_file, sr=None)

    st.audio(uploaded_file, format="audio/wav")

    # 🔊 Waveform
    st.subheader("📈 Audio Waveform")
    fig1, ax1 = plt.subplots()
    ax1.plot(audio)
    ax1.set_title("Waveform")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Amplitude")
    st.pyplot(fig1)

    # MFCC
    mfcc, features = extract_features(audio, sr)

    st.subheader("📊 MFCC Features")
    fig2, ax2 = plt.subplots()
    img = ax2.imshow(mfcc, aspect='auto', origin='lower')
    ax2.set_title("MFCC Spectrogram")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("MFCC Coefficients")
    fig2.colorbar(img)
    st.pyplot(fig2)

    # Prediction
    with st.spinner("🔍 Analyzing audio..."):
        prediction = model.predict([features])

    st.divider()

    if prediction[0] == 1:
        st.error("⚠️ SCAM / PHISHING CALL DETECTED")
        st.markdown("🚨 Do not share OTP, bank details, or passwords.")
    else:
        st.success("✅ NORMAL CALL")
        st.markdown("✔ This call appears to be safe.")

# Footer
st.divider()
st.markdown(
    "<p style='text-align: center; font-size: 12px;'>Final Year Project | AI Voice Phishing Detection</p>",
    unsafe_allow_html=True
)
