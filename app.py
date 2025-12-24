import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import tempfile

# -------------------------------
# Load trained model
# -------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AI Voice Phishing Detector", layout="centered")

st.title("📞 AI Voice Phishing Detector")
st.write("Upload a call recording to detect whether it is **Phishing** or **Normal**.")

# -------------------------------
# Audio Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload WAV audio file",
    type=["wav"]
)

# -------------------------------
# MFCC Extraction Function
# -------------------------------
def extract_mfcc(audio_path):
    y, sr = librosa.load(audio_path, duration=5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc

# -------------------------------
# When file uploaded
# -------------------------------
if uploaded_file is not None:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(uploaded_file)

    # -------------------------------
    # MFCC Extraction
    # -------------------------------
    mfcc = extract_mfcc(audio_path)

    # -------------------------------
    # MFCC Graph
    # -------------------------------
    st.subheader("🎵 MFCC Feature Visualization")

    fig1, ax1 = plt.subplots()
    img = librosa.display.specshow(
        mfcc,
        x_axis="time",
        ax=ax1
    )
    ax1.set(title="MFCC Heatmap")
    fig1.colorbar(img, ax=ax1)

    st.pyplot(fig1)

    # -------------------------------
    # Prepare Features for Model
    # -------------------------------
    features = np.mean(mfcc.T, axis=0)

    # -------------------------------
    # Prediction
    # -------------------------------
    prediction = model.predict([features])
    probability = model.predict_proba([features])[0]

    # -------------------------------
    # Result Output
    # -------------------------------
    st.subheader("🔍 Detection Result")

    if prediction[0] == 1:
        st.error("⚠️ PHISHING / SCAM CALL DETECTED")
    else:
        st.success("✅ NORMAL CALL")

    # -------------------------------
    # Confidence Graph
    # -------------------------------
    st.subheader("📊 Prediction Confidence")

    fig2, ax2 = plt.subplots()
    labels = ["Normal", "Phishing"]
    ax2.bar(labels, probability)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Probability")

    st.pyplot(fig2)

    # Clean temp file
    os.remove(audio_path)
