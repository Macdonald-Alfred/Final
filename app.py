import os
import warnings
import numpy as np
import librosa
import joblib
from flask import Flask, render_template, request, redirect

warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

# Load the model
model_path = 'C:/Users/User/Desktop/mmm/Sound.pkl'  # Adjust the model path accordingly
model = joblib.load(model_path)

# Define the feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=5.0)

    # Extract audio features
    length = librosa.get_duration(y=y, sr=sr)
    chroma_stft_mean = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)[0]
    rms_mean = np.mean(librosa.feature.rms(y=y), axis=1)[0]
    rms_var = np.var(librosa.feature.rms(y=y), axis=1)[0]
    spectral_centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr), axis=1)[0]
    spectral_bandwidth_mean = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr), axis=1)[0]
    zero_crossing_rate_mean = np.mean(librosa.feature.zero_crossing_rate(y=y), axis=1)[0]
    mfcc_mean = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1), axis=1)[0]

    features = np.array([
        length, chroma_stft_mean, rms_mean, rms_var,
        spectral_centroid_mean, spectral_bandwidth_mean,
        zero_crossing_rate_mean, mfcc_mean
    ])

    return features

# Predict function
def predict_audio(file_path):
    features = extract_features(file_path).reshape(1, -1)  # Reshape for the model
    prediction = model.predict(features)
    return prediction[0]  # Return the first element of the prediction

# Main route
@app.route("/", methods=['POST', 'GET'])
def index():
    result = None  # Initialize result variable
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            # Create uploads directory if it doesn't exist
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)

            # Save file temporarily
            file_path = os.path.join(upload_dir, file.filename)
            file.save(file_path)
            result = predict_audio(file_path)  # Get prediction
            os.remove(file_path)  # Clean up after prediction
        return render_template('ui.html', request="POST", result=result)
    else:
        return render_template("ui.html")

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
