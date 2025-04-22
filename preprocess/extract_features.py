import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

def load_training_data(file_path):
    """Load training data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        # Convert outcomes to binary labels (1 for Abnormal, 0 for Normal)
        df['label'] = df['Outcome'].map({'Abnormal': 1, 'Normal': 0})
        # Create dictionary mapping patient IDs to labels
        patient_labels = dict(zip(df['Patient ID'], df['label']))
        return patient_labels
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None

def extract_features(audio_file):
    """Extract features from audio file."""
    try:
        # Load audio file with a fixed sample rate
        y, sr = librosa.load(audio_file, sr=4000)  # Using 4kHz sample rate
        
        # Extract features
        features = []
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        
        # Spectral features with adjusted parameters
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, fmin=50, n_bands=6)
        
        features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
        features.extend([np.mean(spectral_contrast), np.std(spectral_contrast)])
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features.extend([np.mean(zcr), np.std(zcr)])
        
        # RMS energy
        rms = librosa.feature.rms(y=y)
        features.extend([np.mean(rms), np.std(rms)])
        
        return features
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

def process_dataset(audio_dir, patient_labels, output_file):
    """Process all audio files and save features to CSV."""
    records = []
    
    # Get list of audio files
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            # Extract patient ID from filename
            patient_id = int(audio_file.split('_')[0])
            
            # Get label for this patient
            if patient_id in patient_labels:
                label = patient_labels[patient_id]
                
                # Extract features
                features = extract_features(os.path.join(audio_dir, audio_file))
                
                if features is not None:
                    # Add patient ID and label to features
                    record = [patient_id] + features + [label]
                    records.append(record)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
    
    if records:
        # Create column names
        feature_names = []
        for i in range(13):  # MFCCs
            feature_names.extend([f'mfcc{i+1}_mean', f'mfcc{i+1}_std'])
        feature_names.extend(['spectral_centroid_mean', 'spectral_centroid_std',
                            'spectral_rolloff_mean', 'spectral_rolloff_std',
                            'spectral_contrast_mean', 'spectral_contrast_std',
                            'zcr_mean', 'zcr_std',
                            'rms_mean', 'rms_std'])
        
        columns = ['patient_id'] + feature_names + ['label']
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(records, columns=columns)
        df.to_csv(output_file, index=False)
        print(f"Saved features to {output_file}")
        print(f"Processed {len(df)} records successfully")
    else:
        print("No valid features were extracted")

if __name__ == "__main__":
    # Load training data
    training_data_path = "data/training_data.csv"
    patient_labels = load_training_data(training_data_path)
    
    if patient_labels is not None:
        # Process dataset
        audio_dir = "data/records/"
        output_file = "data/processed_features.csv"
        process_dataset(audio_dir, patient_labels, output_file)
    else:
        print("Failed to load training data. Exiting.")
