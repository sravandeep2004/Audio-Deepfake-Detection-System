from flask import Flask, render_template, request
import numpy as np
import librosa
import librosa.display
import joblib
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pydub import AudioSegment

matplotlib.use('Agg')  

app = Flask(__name__)

try:
    model = joblib.load("model.pkl")  
    scaler = joblib.load("scaler.pkl")  
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model, scaler = None, None

test_folder = "Test"
spectrogram_folder = "static/spectrograms"

os.makedirs(test_folder, exist_ok=True)
os.makedirs(spectrogram_folder, exist_ok=True)

def convert_to_wav(file_path):
    if file_path.lower().endswith(".mp3"):
        try:
            sound = AudioSegment.from_mp3(file_path)
            wav_path = file_path.replace(".mp3", ".wav")
            sound.export(wav_path, format="wav")
            return wav_path
        except Exception as e:
            print(f"Error converting MP3 to WAV: {e}")
            return file_path
    return file_path

def generate_spectrogram(file_path, output_path):
    try:
        X, sr = librosa.load(file_path, sr=16000, mono=True)
        plt.figure(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y=X, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.savefig(output_path)
        plt.close()
    except Exception as e:
        print(f"Error generating spectrogram: {e}")

def extract_features(file_path):
    try:
        X, sr = librosa.load(file_path, sr=16000, mono=True)

        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=X, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr).T, axis=0)
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=X, sr=sr).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=X).T, axis=0)

        return np.concatenate((mfccs, spectral_contrast, tonnetz, rolloff, rms))
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def model_page():
    spectrogram_path = None
    result_label = None

    if request.method == "POST":
        selected_file = request.files.get("audio_file")
        if not selected_file:
            return render_template("model.html", error="No file uploaded")

        file_name = selected_file.filename
        file_path = os.path.join(test_folder, file_name)
        selected_file.save(file_path)

        file_path = convert_to_wav(file_path)

        spectrogram_path = os.path.join(spectrogram_folder, f"{file_name}.png")
        generate_spectrogram(file_path, spectrogram_path)

        features = extract_features(file_path)
        if features is None or model is None or scaler is None:
            return render_template("model.html", error="Error processing audio file.")

        features_df = pd.DataFrame([features])

        features_scaled = scaler.transform(features_df)

        prediction = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled)[0]

        file_label = f"File: {file_name}"
        if prediction == 1:
            result_label = f"<span class='fake-result'>Fake with {confidence[1] * 100:.2f}% confidence</span>"
        else:
            result_label = f"<span class='real-result'>Real with {confidence[0] * 100:.2f}% confidence</span>"

        return render_template("model.html", file_label=file_label, result_label=result_label, spectrogram_path=spectrogram_path)

    return render_template("model.html")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
