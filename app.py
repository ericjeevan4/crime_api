import os
import gdown
import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Step 1: Download model from Google Drive if not already present
MODEL_FILE = "crime_model.pkl"
GOOGLE_DRIVE_FILE_ID = "1iH9JsBwMPkHV_Rd0W-91-LGS2QRB6k6c"
MODEL_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"

if not os.path.exists(MODEL_FILE):
    print("🔽 Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_FILE, quiet=False)

# Step 2: Load the model
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# Step 3: Define predict route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    try:
        features = np.array([
            data['Pincode'],
            data['Latitude'],
            data['Longitude'],
            data['Zone_Name']
        ]).reshape(1, -1)

        prediction = model.predict(features)
        return jsonify({"prediction": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Step 4: Health check route
@app.route("/")
def home():
    return "🚀 Crime Prediction API is up and running!"

# Step 5: Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
