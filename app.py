from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import gdown

app = Flask(__name__)

# Define path and Google Drive direct download ID
MODEL_PATH = "crime_model.pkl"
DRIVE_URL = "https://drive.google.com/uc?export=download&id=1iH9JsBwMPkHV_Rd0W-91-LGS2QRB6k6c"

# Download model from Google Drive if not already present
if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model from Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# Load the trained model
print("📦 Loading crime prediction model...")
model = joblib.load(MODEL_PATH)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # JSON input from user
        input_df = pd.DataFrame([data])  # Convert to DataFrame
        prediction = model.predict(input_df)[0]  # Make prediction
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

# Health check route
@app.route('/')
def home():
    return "✅ Crime Prediction API is live!"

# Run locally (will be handled by gunicorn on Render)
if __name__ == '__main__':
    app.run(debug=True)
