import os
import librosa
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

print("Training started...")

X = []
y = []

dataset_path = "dataset"

folders = {
    "normal_calls": 0,
    "phishing_calls": 1
}

for folder, label in folders.items():
    folder_path = os.path.join(dataset_path, folder)

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)

            audio, sr = librosa.load(file_path, duration=5)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc.T, axis=0)

            X.append(mfcc_mean)
            y.append(label)

X = np.array(X)
y = np.array(y)

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

with open("voice_phishing_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained & saved successfully")
