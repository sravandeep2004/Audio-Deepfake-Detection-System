import os
import librosa
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import librosa.effects

dataset_path = "Test"
output_csv = "dataset.csv"
labels = {"DeepfakeExample": 1, "RealExample": 0}

def extract_features(file_path):
    X, sr = librosa.load(file_path, sr=16000, mono=True)

    mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)

    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=X, sr=sr).T, axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr).T, axis=0)

    rolloff = np.mean(librosa.feature.spectral_rolloff(y=X, sr=sr).T, axis=0)

    rms = np.mean(librosa.feature.rms(y=X).T, axis=0)

    final_features = np.concatenate((mfccs, spectral_contrast, tonnetz, rolloff, rms))

    print(f"Extracted {len(final_features)} features from {file_path}")  # Debugging

    return final_features


def augment_audio(file_path):
    X, sr = librosa.load(file_path, sr=16000, mono=True)
    
    X_pitch = librosa.effects.pitch_shift(X, sr=sr, n_steps=2)

    X_stretch = librosa.effects.time_stretch(X, rate=1.2)

    return X_pitch, X_stretch

data = []
for file_name in os.listdir(dataset_path):
    if file_name.endswith(".wav"):
        file_path = os.path.join(dataset_path, file_name)
        features = extract_features(file_path)
        label = labels["DeepfakeExample"] if "DeepfakeExample" in file_name else labels["RealExample"]
        data.append(np.append(features, label))

        X_pitch, X_stretch = augment_audio(file_path)
        pitch_features = extract_features(file_path) 
        stretch_features = extract_features(file_path)

        data.append(np.append(pitch_features, label))
        data.append(np.append(stretch_features, label))

df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print("Dataset saved successfully!")

df = pd.read_csv("dataset.csv")

X = df.iloc[:, :-1]  
y = df.iloc[:, -1]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=5, subsample=0.7)
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

test_accuracy = model.score(X_test, y_test)
print(f"Test Accuracy on New Data: {test_accuracy * 100:.2f}%")