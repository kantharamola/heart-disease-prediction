import librosa
import numpy as np
import pandas as pd
import os

# Load training data to get labels
label_data = pd.read_csv("data/training_data.csv", delimiter='\t')

# Convert "Outcome" to a binary label (1 = Abnormal, 0 = Normal)
label_data["label"] = label_data["Outcome"].apply(lambda x: 1 if x == "Abnormal" else 0)

# Create a dictionary mapping patient IDs to labels
label_dict = dict(zip(label_data["Patient ID"].astype(str), label_data["label"]))

def extract_features(audio_file):
    """Extracts MFCC, Chroma, and Spectral Contrast features from a heart sound file."""
    try:
        y, sr = librosa.load(audio_file, sr=4000)  # Load audio at 4kHz
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, fmin=50)

        features = np.concatenate((
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(spectral_contrast, axis=1)
        ))
        return features
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None  # Return None for failed feature extraction


def process_dataset(data_folder):
    """Processes all heart sound files and generates processed_features.csv."""
    records = []
    for file in os.listdir(data_folder):
        if file.endswith(".wav"):
            patient_id = file.split("_")[0]  # Extract ID from filename
            
            # Get label from training_data.csv
            if patient_id not in label_dict:
                print(f"⚠ No label found for {patient_id}. Skipping.")
                continue
            
            features = extract_features(os.path.join(data_folder, file))
            if features is None:
                print(f"⚠ Skipping {file} due to feature extraction error.")
                continue
            
            label = label_dict[patient_id]  # Get label from dictionary
            records.append([patient_id, *features, label])

    if records:
        columns = ["patient_id"] + [f"feat_{i}" for i in range(len(records[0]) - 2)] + ["label"]
        df = pd.DataFrame(records, columns=columns)
        df.dropna(inplace=True)  # Drop NaN values
        df.to_csv("data/processed_features.csv", index=False)
        print(f"Feature extraction complete. Processed {len(df)} records.")
    else:
        print("⚠ No valid features extracted. Check the dataset!")

if __name__ == "__main__":
    process_dataset("data/records/")
