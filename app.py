import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load Model & Scaler
model = tf.keras.models.load_model("models/heart_disease_model.h5")
scaler = joblib.load("models/scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    patient_id = request.form['patient_id']
    
    # Load data
    data = pd.read_csv("data/processed_features.csv")
    
    # Get patient data
    patient_data = data[data['patient_id'] == int(patient_id)]
    
    if patient_data.empty:
        return jsonify({"error": "Patient not found"})
    
    # Extract Features (Exclude `patient_id` and `label`)
    X = patient_data.iloc[:, 1:-1].values  # Ensuring 32 features only
    
    # 🚨 Debugging Step: Print Feature Count Before Scaling
    print("Feature count before scaling:", X.shape[1])  # Should print 32

    if X.shape[1] != 32:
        return jsonify({"error": "Incorrect number of features selected!"})

    # Normalize Features
    X_scaled = scaler.transform(X)  # Ensure the scaler matches these 32 features

    # 🚨 Debugging Step: Print Shape Before Reshaping
    print("Shape before reshaping:", X_scaled.shape)  # Should be (1, 32)

    if X_scaled.shape[1] != 32:
        return jsonify({"error": "Feature scaling altered dimensions unexpectedly!"})

    # Reshape Correctly
    X_reshaped = X_scaled.reshape(1, 32, 1)  # Ensure model expects (batch, 32, 1)

    # 🚨 Debugging Step: Print Final Shape Before Prediction
    print("Final input shape for model:", X_reshaped.shape)  

    # Make Prediction
    prediction = model.predict(X_reshaped)[0][0]

    # Risk Classification
    advice = "High Risk" if prediction > 0.5 else "Low Risk"

    return jsonify({
        "patient_id": patient_id, 
        "advice": advice, 
        "risk_score": float(prediction)
    })

if __name__ == '__main__':
    app.run(debug=True)
