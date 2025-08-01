from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import gdown

app = Flask(__name__)

# Google Drive direct download link (make sure 'Anyone with link' is enabled)
MODEL_PATH = "crime_model.pkl"
DRIVE_URL = "https://drive.google.com/uc?export=download&id=1iH9JsBwMPkHV_Rd0W-91-LGS2QRB6k6c"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model from Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# Load the ML model
print("📦 Loading crime prediction model...")
model = joblib.load(MODEL_PATH)

# Home route
@app.route('/')
def home():
    return "✅ Crime Prediction API is live!"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]

        return jsonify({
            "prediction": prediction
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
