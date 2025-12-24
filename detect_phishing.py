import sys
import librosa
import numpy as np
import pickle
import os

print("Detector started")

# Check argument
if len(sys.argv) < 2:
    print("Usage: python detect_phishing.py <audio_file.wav>")
    sys.exit()

audio_path = sys.argv[1]

# Check file exists
if not os.path.exists(audio_path):
    print("Audio file not found!")
    sys.exit()

print(f"Processing: {audio_path}")

# Load model
with open("voice_phishing_model.pkl", "rb") as f:
    model = pickle.load(f)

# Extract features
y, sr = librosa.load(audio_path, duration=5)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_mean = np.mean(mfcc.T, axis=0)
mfcc_mean = mfcc_mean.reshape(1, -1)

# Prediction
prediction = model.predict(mfcc_mean)

if prediction[0] == 1:
    print("⚠️ SCAM / PHISHING CALL DETECTED")
else:
    print("✅ NORMAL CALL")
