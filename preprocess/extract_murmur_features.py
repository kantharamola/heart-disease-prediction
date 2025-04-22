import pandas as pd
import numpy as np

def extract_murmur_features(input_file, output_file):
    """Extract murmur-related features from training data."""
    try:
        # Read the training data with proper parsing
        df = pd.read_csv(input_file, sep=',', skipinitialspace=True)
        
        # Print column names for debugging
        print("Available columns:", df.columns.tolist())
        
        # Select murmur-related columns that exist in the data
        murmur_columns = [
            'Murmur', 'Murmur locations', 'Most audible location',
            'Systolic murmur timing', 'Systolic murmur shape', 'Systolic murmur grading',
            'Systolic murmur pitch', 'Systolic murmur quality'
        ]
        
        # Add diastolic columns if they exist
        diastolic_columns = [
            'Diastolic murmur timing', 'Diastolic murmur shape', 'Diastolic murmur grading',
            'Diastolic murmur pitch', 'Diastolic murmur quality'
        ]
        
        # Filter to only include columns that exist
        murmur_columns = [col for col in murmur_columns if col in df.columns]
        diastolic_columns = [col for col in diastolic_columns if col in df.columns]
        
        # Create a new DataFrame with murmur features
        murmur_df = df[['Patient ID'] + murmur_columns + diastolic_columns + ['Outcome']].copy()
        
        # Convert categorical features to numerical values
        # Murmur presence
        murmur_df['Murmur_present'] = murmur_df['Murmur'].map({
            'Present': 1,
            'Absent': 0,
            'Unknown': 0.5
        })
        
        # Murmur locations (count of locations)
        murmur_df['Murmur_locations_count'] = murmur_df['Murmur locations'].apply(
            lambda x: len(str(x).split('+')) if pd.notna(x) and x != 'nan' else 0
        )
        
        # Most audible location (one-hot encoding)
        locations = ['AV', 'PV', 'TV', 'MV']
        for loc in locations:
            murmur_df[f'Most_audible_{loc}'] = murmur_df['Most audible location'].apply(
                lambda x: 1 if pd.notna(x) and loc in str(x) else 0
            )
        
        # Systolic murmur features
        systolic_timing = ['Early-systolic', 'Mid-systolic', 'Holosystolic']
        systolic_shape = ['Plateau', 'Diamond', 'Decrescendo']
        systolic_grading = ['I/VI', 'II/VI', 'III/VI', 'IV/VI', 'V/VI', 'VI/VI']
        systolic_pitch = ['Low', 'Medium', 'High']
        systolic_quality = ['Harsh', 'Blowing', 'Musical']
        
        # Create one-hot encoded features for systolic characteristics
        for timing in systolic_timing:
            murmur_df[f'Systolic_timing_{timing}'] = murmur_df['Systolic murmur timing'].apply(
                lambda x: 1 if pd.notna(x) and timing in str(x) else 0
            )
        
        for shape in systolic_shape:
            murmur_df[f'Systolic_shape_{shape}'] = murmur_df['Systolic murmur shape'].apply(
                lambda x: 1 if pd.notna(x) and shape in str(x) else 0
            )
        
        for grade in systolic_grading:
            murmur_df[f'Systolic_grade_{grade}'] = murmur_df['Systolic murmur grading'].apply(
                lambda x: 1 if pd.notna(x) and grade in str(x) else 0
            )
        
        for pitch in systolic_pitch:
            murmur_df[f'Systolic_pitch_{pitch}'] = murmur_df['Systolic murmur pitch'].apply(
                lambda x: 1 if pd.notna(x) and pitch in str(x) else 0
            )
        
        for quality in systolic_quality:
            murmur_df[f'Systolic_quality_{quality}'] = murmur_df['Systolic murmur quality'].apply(
                lambda x: 1 if pd.notna(x) and quality in str(x) else 0
            )
        
        # Diastolic murmur features (only if columns exist)
        if diastolic_columns:
            diastolic_timing = ['Early-diastolic', 'Mid-diastolic', 'Holodiastolic']
            diastolic_shape = ['Plateau', 'Diamond', 'Decrescendo']
            diastolic_grading = ['I/IV', 'II/IV', 'III/IV', 'IV/IV']
            diastolic_pitch = ['Low', 'Medium', 'High']
            diastolic_quality = ['Harsh', 'Blowing', 'Musical']
            
            # Create one-hot encoded features for diastolic characteristics
            for timing in diastolic_timing:
                murmur_df[f'Diastolic_timing_{timing}'] = murmur_df['Diastolic murmur timing'].apply(
                    lambda x: 1 if pd.notna(x) and timing in str(x) else 0
                )
            
            for shape in diastolic_shape:
                murmur_df[f'Diastolic_shape_{shape}'] = murmur_df['Diastolic murmur shape'].apply(
                    lambda x: 1 if pd.notna(x) and shape in str(x) else 0
                )
            
            for grade in diastolic_grading:
                murmur_df[f'Diastolic_grade_{grade}'] = murmur_df['Diastolic murmur grading'].apply(
                    lambda x: 1 if pd.notna(x) and grade in str(x) else 0
                )
            
            for pitch in diastolic_pitch:
                murmur_df[f'Diastolic_pitch_{pitch}'] = murmur_df['Diastolic murmur pitch'].apply(
                    lambda x: 1 if pd.notna(x) and pitch in str(x) else 0
                )
            
            for quality in diastolic_quality:
                murmur_df[f'Diastolic_quality_{quality}'] = murmur_df['Diastolic murmur quality'].apply(
                    lambda x: 1 if pd.notna(x) and quality in str(x) else 0
                )
        
        # Convert outcome to binary
        murmur_df['label'] = murmur_df['Outcome'].map({'Abnormal': 1, 'Normal': 0})
        
        # Drop original columns
        columns_to_drop = murmur_columns + diastolic_columns + ['Outcome']
        murmur_df = murmur_df.drop(columns=columns_to_drop)
        
        # Save to CSV
        murmur_df.to_csv(output_file, index=False)
        print(f"Saved murmur features to {output_file}")
        print(f"Processed {len(murmur_df)} records")
        
        return murmur_df
        
    except Exception as e:
        print(f"Error processing murmur features: {e}")
        return None

if __name__ == "__main__":
    input_file = "data/training_data.csv"
    output_file = "data/murmur_features.csv"
    extract_murmur_features(input_file, output_file) 