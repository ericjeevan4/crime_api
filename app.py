from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("chennai_crime_predictor.joblib")

# Define the output features
output_features = [
    'Crime_Type', 'Crime_Subtype', 'Crime_Severity', 'Victim_Age_Group',
    'Victim_Gender', 'Suspect_Count', 'Weapon_Used', 'Gang_Involvement',
    'Vehicle_Used', 'CCTV_Captured', 'Reported_By', 'Response_Time_Minutes',
    'Arrest_Made', 'Crime_History_Count', 'Crimes_Same_Type_Count', 'Risk_Level'
]

@app.route('/')
def home():
    return jsonify({"message": "🚀 Chennai Crime Predictor API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        input_df = pd.DataFrame([{
            "Area_Name": data['area_name'],
            "Pincode": data['pincode'],
            "Latitude": data['latitude'],
            "Longitude": data['longitude'],
            "Zone_Name": data['zone_name']
        }])

        predictions = model.predict(input_df)
        result = {output_features[i]: predictions[0][i] for i in range(len(output_features))}
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
