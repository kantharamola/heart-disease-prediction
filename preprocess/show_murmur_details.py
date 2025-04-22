import pandas as pd

def display_patient_murmur_details(patient_data):
    """Display murmur details for a patient in a UI-like format."""
    print("\n🫀 CARDIOVASCULAR METRICS")
    print("========================")
    
    print("\nMurmur Assessment")
    print("----------------")
    print(f"Presence:           {patient_data.get('Murmur', 'Unknown')}")
    print(f"Location:           {patient_data.get('Murmur locations', 'Unknown')}")
    print(f"Most Audible:       {patient_data.get('Most audible location', 'Unknown')}")
    
    print("\nSystolic Details")
    print("---------------")
    print(f"Grade:             {patient_data.get('Systolic murmur grading', 'Unknown')}")
    print(f"Pitch:             {patient_data.get('Systolic murmur pitch', 'Unknown')}")
    print(f"Quality:           {patient_data.get('Systolic murmur quality', 'Unknown')}")
    print(f"Shape:             {patient_data.get('Systolic murmur shape', 'Unknown')}")
    print(f"Timing:            {patient_data.get('Systolic murmur timing', 'Unknown')}")
    
    print("\nDiastolic Details")
    print("----------------")
    print(f"Grade:             {patient_data.get('Diastolic murmur grading', 'Unknown')}")
    print(f"Pitch:             {patient_data.get('Diastolic murmur pitch', 'Unknown')}")
    print(f"Quality:           {patient_data.get('Diastolic murmur quality', 'Unknown')}")
    print(f"Shape:             {patient_data.get('Diastolic murmur shape', 'Unknown')}")
    print(f"Timing:            {patient_data.get('Diastolic murmur timing', 'Unknown')}")
    
    print("\n👤 PATIENT DEMOGRAPHICS")
    print("=====================")
    print(f"Age:               {patient_data.get('Age', 'Unknown')}")
    print(f"Sex:               {patient_data.get('Sex', 'Unknown')}")
    print(f"Height:            {patient_data.get('Height', 'Unknown')} cm")
    print(f"Weight:            {patient_data.get('Weight', 'Unknown')} kg")
    print(f"Pregnancy Status:  {patient_data.get('Pregnancy status', 'Unknown')}")
    
    print("\n⚠️ OUTCOME")
    print("=========")
    print(f"Diagnosis:         {patient_data.get('Outcome', 'Unknown')}")

def show_murmur_details(input_file, patient_id=None):
    """Display murmur details from training data."""
    try:
        # Read the training data
        df = pd.read_csv(input_file)
        
        if patient_id is not None:
            # Display details for specific patient
            patient_data = df[df['Patient ID'] == patient_id].iloc[0].to_dict()
            print(f"\nDetails for Patient ID: {patient_id}")
            print("=" * 30)
            display_patient_murmur_details(patient_data)
            return
        
        # If no patient ID specified, show summary of all records
        print(f"\nTotal Records: {len(df)}")
        print("\nOutcome Distribution:")
        print("-" * 20)
        outcome_counts = df['Outcome'].value_counts()
        for outcome, count in outcome_counts.items():
            print(f"{outcome}: {count} ({count/len(df)*100:.1f}%)")
        
        print("\nMurmur Presence Distribution:")
        print("-" * 20)
        murmur_counts = df['Murmur'].value_counts()
        for status, count in murmur_counts.items():
            print(f"{status}: {count} ({count/len(df)*100:.1f}%)")
        
        # Save detailed data to CSV
        output_file = "data/murmur_details.csv"
        df.to_csv(output_file, index=False)
        print(f"\nDetailed murmur data saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing murmur details: {e}")
        return None

if __name__ == "__main__":
    input_file = "data/training_data.csv"
    # Example: Show details for patient ID 36327 (from your screenshot)
    show_murmur_details(input_file, patient_id=36327) 