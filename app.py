import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded_file is not None:
    # librosa can read file-like object directly
    y, sr = librosa.load(uploaded_file, sr=None)
    st.audio(uploaded_file)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title("MFCC Visualization")
    plt.tight_layout()
    st.pyplot(plt)


        

