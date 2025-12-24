import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ================================
# STEP 1: LOAD AUDIO FILE
# ================================

audio_path = "sample_call.wav"   # <-- change if needed

y, sr = librosa.load(audio_path, sr=None)

print("Audio Loaded Successfully")
print("Sample Rate:", sr)

# ================================
# STEP 2: MFCC FEATURE EXTRACTION
# ================================

mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Mean MFCC (used for ML model)
mfcc_mean = np.mean(mfcc.T, axis=0)

# ================================
# STEP 3: MFCC VISUALIZATION
# ================================

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time', sr=sr)
plt.colorbar(format="%+2.0f dB")
plt.title("MFCC Feature Visualization")
plt.xlabel("Time")
plt.ylabel("MFCC Coefficients")
plt.tight_layout()
plt.show()

# ================================
# STEP 4: SPECTROGRAM VISUALIZATION
# ================================

stft = np.abs(librosa.stft(y))
spectrogram = librosa.amplitude_to_db(stft, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.title("Frequency Spectrum (Spectrogram)")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ================================
# STEP 5: TRAIN SIMPLE MODEL (Demo)
# ================================

# Dummy dataset (replace with real dataset if available)
X = np.array([
    mfcc_mean,
    mfcc_mean + 1,
    mfcc_mean - 1
])

y_labels = np.array([1, 0, 1])  # 1 = phishing, 0 = normal

X_train, X_test, y_train, y_test = train_test_split(
    X, y_labels, test_size=0.3, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ================================
# STEP 6: PREDICTION
# ================================

prediction = model.predict([mfcc_mean])

print("\n========== RESULT ==========")
if prediction[0] == 1:
    print("🚨 PHISHING CALL DETECTED")
else:
    print("✅ NORMAL CALL")

# ================================
# END
# ================================

        
