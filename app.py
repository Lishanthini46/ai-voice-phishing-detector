import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import pickle

# -----------------------------
# Load trained model
# -----------------------------
model = pickle.load(open("voice_phishing_model.pkl", "rb"))

# -----------------------------
# Feature extraction function
# -----------------------------
def extract_mfcc(audio_path):
    y, sr = librosa.load(audio_path, duration=5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc, mfcc_mean


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Voice Phishing Detector", layout="centered")

st.title("🎙️ AI Voice Phishing Detector")
st.write("Upload an audio file to detect whether it is **Phishing** or **Safe**.")

uploaded_file = st.file_uploader(
    "Upload WAV audio file",
    type=["wav"]
)

# -----------------------------
# When file is uploaded
# -----------------------------
if uploaded_file is not None:

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    # Play audio
    st.audio(uploaded_file)

    # Extract MFCC
    mfcc, mfcc_mean = extract_mfcc(audio_path)

    # -----------------------------
    # MFCC Visualization
    # -----------------------------
    st.subheader("🎵 MFCC Feature Visualization")

    fig, ax = plt.subplots()
    img = librosa.display.specshow(
        mfcc,
        x_axis="time",
        ax=ax
    )
    fig.colorbar(img, ax=ax)
    st.pyplot(fig)

    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict([mfcc_mean])

    st.subheader("🔍 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Phishing Voice Detected")
    else:
        st.success("✅ Safe Voice")


