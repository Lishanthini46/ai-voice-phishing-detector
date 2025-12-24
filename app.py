import streamlit as st
import numpy as np
import tempfile
import pickle
import matplotlib.pyplot as plt
import librosa
import librosa.display

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
    return y, sr, mfcc, mfcc_mean

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

    # Extract MFCC + Audio
    y, sr, mfcc, mfcc_mean = extract_mfcc(audio_path)

    # -----------------------------
    # MFCC Visualization
    # -----------------------------
    st.subheader("🎵 MFCC Feature Visualization")
    mfcc_db = librosa.power_to_db(np.abs(mfcc), ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(
        mfcc_db,
        aspect="auto",
        origin="lower",
        cmap="viridis"
    )
    ax.set_title("MFCC (Mel-Frequency Cepstral Coefficients)")
    ax.set_xlabel("Time")
    ax.set_ylabel("MFCC Coefficients")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)

    # -----------------------------
    # Frequency Visualization
    # -----------------------------
    st.subheader("📈 Frequency Spectrum")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    D = np.abs(librosa.stft(y))
    librosa.display.specshow(
        librosa.amplitude_to_db(D, ref=np.max),
        sr=sr,
        y_axis='log',
        x_axis='time',
        cmap='magma',
        ax=ax2
    )
    ax2.set_title("Frequency Spectrum")
    fig2.colorbar(ax2.images[0], ax=ax2, format="%+2.0f dB")
    st.pyplot(fig2)

    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict([mfcc_mean])

    st.subheader("🔍 Prediction Result")
    if prediction[0] == 1:
        st.error("⚠️ Phishing Voice Detected")
    else:
        st.success("✅ Safe Voice")



    

   

