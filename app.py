import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify
import gdown

app = Flask(__name__)

MODEL_PATH = "chennai_crime_predictor.pkl"
MODEL_DRIVE_ID = "1iH9JsBwMPkHV_Rd0W-91-LGS2QRB6k6c"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("⏬ Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}", MODEL_PATH, quiet=False)

# Load the model
model = joblib.load(MODEL_PATH)

# Input and output columns
input_columns = ['Area_Name', 'Pincode', 'Latitude', 'Longitude', 'Zone_Name']
output_columns = [
    'Crime_Type', 'Crime_Subtype', 'Crime_Severity', 'Victim_Age_Group', 
    'Victim_Gender', 'Suspect_Count', 'Weapon_Used', 'Gang_Involvement', 
    'Vehicle_Used', 'CCTV_Captured', 'Reported_By', 'Response_Time_Minutes',
    'Arrest_Made', 'Crime_History_Count', 'Crimes_Same_Type_Count', 'Risk_Level'
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = pd.DataFrame([{
        'Area_Name': data['area_name'],
        'Pincode': data['pincode'],
        'Latitude': data['latitude'],
        'Longitude': data['longitude'],
        'Zone_Name': data['zone_name']
    }])
    prediction = model.predict(input_data)[0]
    result = {key: value for key, value in zip(output_columns, prediction)}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
