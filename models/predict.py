import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import roc_curve, accuracy_score

# Load the trained model and scaler
def load_model():
    model_path = "models/heart_disease_model.h5"
    scaler_path = "models/scaler.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Determine optimal decision threshold using ROC curve
def find_optimal_threshold(model, scaler):
    df = pd.read_csv("D:\heart_diseasee\heart_disease\data\processed_features.csv")

    if 'patient_id' in df.columns:
        df = df.drop(columns=['patient_id'])

    X_test = df.iloc[:, :-1].values
    y_test = df.iloc[:, -1].values

    expected_features = scaler.n_features_in_
    if X_test.shape[1] != expected_features:
        X_test = X_test[:, -expected_features:]

    X_test = scaler.transform(X_test)
    X_test = np.expand_dims(X_test, axis=-1)

    y_probs = model.predict(X_test).flatten()

    fpr, tpr, thresholds = roc_curve(y_test, y_probs)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx] if 0 < optimal_idx < len(thresholds) else 0.5

    print(f"Optimal Decision Threshold: {optimal_threshold:.4f}")
    return optimal_threshold

# Prediction function
def predict(features, threshold):
    model, scaler = load_model()
    
    feature_array = np.array(features).reshape(1, -1)

    expected_features = scaler.n_features_in_
    if feature_array.shape[1] != expected_features:
        feature_array = feature_array[:, -expected_features:]

    feature_array = scaler.transform(feature_array)
    feature_array = np.expand_dims(feature_array, axis=-1)

    prediction = model.predict(feature_array)[0][0]
    print(f"Raw Prediction Score: {prediction:.4f}, Threshold: {threshold:.4f}")

    return int(prediction > threshold)

# Function to evaluate model accuracy
def evaluate_model(threshold):
    model, scaler = load_model()
    
    df = pd.read_csv("D:\heart_diseasee\heart_disease\data\processed_features.csv")

    if 'patient_id' in df.columns:
        df = df.drop(columns=['patient_id'])

    X_test = df.iloc[:, :-1].values
    y_test = df.iloc[:, -1].values

    expected_features = scaler.n_features_in_
    if X_test.shape[1] != expected_features:
        X_test = X_test[:, -expected_features:]

    X_test = scaler.transform(X_test)
    X_test = np.expand_dims(X_test, axis=-1)

    y_probs = model.predict(X_test).flatten()
    predictions = (y_probs > threshold).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy on Processed Features: {accuracy:.4f}")
    return accuracy

# Example Usage
if __name__ == "__main__":
    model, scaler = load_model()
    threshold = find_optimal_threshold(model, scaler)

    df = pd.read_csv("D:\heart_diseasee\heart_disease\data\processed_features.csv")


    feature_columns = df.columns[1:-1]  # Adjust based on your dataset

# Load CSV properly
    df = pd.read_csv("D:\heart_diseasee\heart_disease\data\processed_features.csv", dtype={'patient_id': str})

# Ensure ID consistency
    df['patient_id'] = df['patient_id'].astype(str).str.strip()
    patient_id = str(49876)
    sample = df.loc[df['patient_id'] == patient_id, feature_columns].values

# Debugging output
    print(f"Looking for Patient ID: {repr(patient_id)}")
    print(df['patient_id'].apply(repr).unique()[:10])  # Print first 10 IDs in exact format

# Check if patient exists
    if patient_id not in df['patient_id'].values:
        raise ValueError(f"Patient ID {patient_id} not found. Check data type and formatting.")

    result = predict(sample, threshold)
    print("Prediction:", "No Heart Disease Detected" if result else "Heart Disease Detected")


    evaluate_model(threshold)
