import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import speech_recognition as sr

# ==============================
# APP TITLE
# ==============================
st.set_page_config(page_title="AI Voice Phishing Detector", layout="centered")
st.title("🎙️ AI Voice Phishing Detector")

# ==============================
# FILE UPLOAD
# ==============================
uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if uploaded_file is not None:

    # ==============================
    # LOAD AUDIO
    # ==============================
    y, sr_audio = librosa.load(uploaded_file, sr=None)
    st.audio(uploaded_file)

    # ==============================
    # WAVEFORM
    # ==============================
    st.subheader("🔊 Waveform")
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr_audio, ax=ax1)
    ax1.set_title("Audio Waveform")
    st.pyplot(fig1)

    # ==============================
    # MFCC
    # ==============================
    st.subheader("📊 MFCC Feature Visualization")
    mfcc = librosa.feature.mfcc(y=y, sr=sr_audio, n_mfcc=13)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis="time", sr=sr_audio, ax=ax2)
    plt.colorbar(format="%+2.0f dB")
    ax2.set_title("MFCC Features")
    st.pyplot(fig2)

    # ==============================
    # FREQUENCY SPECTRUM
    # ==============================
    st.subheader("📈 Frequency Spectrum")
    stft = np.abs(librosa.stft(y))
    db = librosa.amplitude_to_db(stft, ref=np.max)
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(db, sr=sr_audio, x_axis="time", y_axis="hz", ax=ax3)
    plt.colorbar()
    ax3.set_title("Spectrogram")
    st.pyplot(fig3)

    # ==============================
    # SPEECH TO TEXT
    # ==============================
    st.subheader("🗣️ Speech to Text")
    recognizer = sr.Recognizer()
    with sr.AudioFile(uploaded_file) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        st.success(f"Recognized Text: {text}")
    except:
        text = ""
        st.error("Could not recognize speech")

    # ==============================
    # PHISHING DETECTION LOGIC (THRESHOLD)
    # ==============================
    st.subheader("🛑 Call Detection Result")
    phishing_keywords = [
        "otp", "bank", "account", "verify", "password",
        "urgent", "click", "link", "payment", "card",
        "suspend", "blocked", "kyc"
    ]

    text_lower = text.lower()
    keyword_count = sum(word in text_lower for word in phishing_keywords)

    # Threshold: detect phishing only if 2 or more keywords match
    threshold = 2

    st.info(f"Phishing Keywords Detected: {keyword_count}")

    if keyword_count >= threshold:
        st.error("🚨 PHISHING CALL DETECTED")
    else:
        st.success("✅ NORMAL CALL")


