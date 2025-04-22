import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import tensorflow as tf
from cnn_rnn_model import build_model
import joblib

def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Handling missing values
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(df.iloc[:, 1:-1].values)  # Exclude ID and label columns
    y = df.iloc[:, -1].values  # Last column is the label
    
    # Normalize data for better convergence
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    return X, y, scaler

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load and preprocess data
X, y, scaler = load_data("data/processed_features.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for CNN input
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Build and train model
model = build_model(input_shape=(X_train.shape[1], 1))

# Add EarlyStopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32, callbacks=[early_stopping])

# Save model and preprocessing tools
model.save("models/heart_disease_model.h5")
joblib.dump(scaler, "models/scaler.pkl")
print("Model training complete and saved.")
