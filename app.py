from flask import Flask, request, jsonify
import gdown
import joblib
import os
import numpy as np

app = Flask(__name__)

MODEL_FILE = "chennai_crime_predictor.pkl"
MODEL_URL = "https://drive.google.com/uc?id=1iH9JsBwMPkHV_Rd0W-91-LGS2QRB6k6c"

# Download model if not already downloaded
if not os.path.exists(MODEL_FILE):
    print("📥 Downloading model...")
    gdown.download(MODEL_URL, MODEL_FILE, quiet=False)

# Load the model
print("📦 Loading model...")
model = joblib.load(MODEL_FILE)
print("✅ Model loaded!")

@app.route('/')
def home():
    return "🚨 Chennai Crime Predictor API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Example expected input: list of features in correct order
        input_features = np.array(data['features']).reshape(1, -1)

        prediction = model.predict(input_features)
        return jsonify({
            "prediction": prediction[0]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

