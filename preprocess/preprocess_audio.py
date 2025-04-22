import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

def preprocess_audio(audio_file):
    """Load and preprocess the heart sound recording."""
    y, sr = librosa.load(audio_file, sr=4000)  # Load at 4kHz
    y = librosa.util.normalize(y)  # Normalize amplitude
    y = librosa.effects.preemphasis(y)  # Apply pre-emphasis filter

    return y, sr

def visualize_waveform(audio_file):
    """Plot the waveform of an audio file."""
    y, sr = preprocess_audio(audio_file)
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform of {os.path.basename(audio_file)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

def extract_mfcc(audio_file):
    """Extract MFCC features from an audio file."""
    y, sr = preprocess_audio(audio_file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

if __name__ == "__main__":
    test_audio = "data/records/49876_TV.wav"  # Example audio file
    visualize_waveform(test_audio)
    mfcc_features = extract_mfcc(test_audio)
    print(f"Extracted MFCC Features: {mfcc_features}")


