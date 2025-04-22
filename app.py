import os
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib
from groq import Groq
import json
import math
from dotenv import load_dotenv
import time

# Load environment variables from .env.local file
load_dotenv('.env.local')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
ENV = os.getenv('FLASK_ENV', 'development')
DEBUG = ENV == 'development'
HOST = '127.0.0.1' if ENV == 'development' else '0.0.0.0'
PORT = int(os.getenv('PORT', 5001))

# Initialize Flask app with additional security headers
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline' cdnjs.cloudflare.com; font-src 'self' cdnjs.cloudflare.com"
    return response

# Load data at startup
try:
    DATA_DF = pd.read_csv("data/processed_features.csv", encoding='utf-8')
    DATA_DF.columns = DATA_DF.columns.str.strip()
    DATA_DF['patient_id'] = DATA_DF['patient_id'].astype(str)
    logger.info(f"Data loaded successfully at startup. Shape: {DATA_DF.shape}")
except Exception as e:
    logger.error(f"Error loading processed features data: {str(e)}")
    raise

# Initialize Groq client
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq client initialized successfully")
else:
    logger.warning("GROQ_API_KEY not found in environment variables")
    groq_client = None

# Add these constants at the top of the file
FEATURE_RANGES = {
    'systolic': {
        'min': 85,
        'max': 200,
        'precision': 1,
        'default': 120
    },
    'diastolic': {
        'min': 55,
        'max': 120,
        'precision': 1,
        'default': 80
    },
    'heart_rate': {
        'min': 45,
        'max': 180,
        'precision': 1,
        'default': 75
    },
    'murmur_grade': {
        'min': 1,
        'max': 6,
        'precision': 0,
        'default': 1
    }
}

def normalize_value(value, feature_name):
    """Normalize a value to a reasonable physiological range with appropriate precision"""
    try:
        value = float(value)
        if feature_name in FEATURE_RANGES:
            range_info = FEATURE_RANGES[feature_name]
            
            # If value is clearly invalid, use the default value
            if value < -1000 or value > 1000 or math.isnan(value):
                value = range_info['default']
            
            # Clamp the value to the defined range
            value = min(range_info['max'], max(range_info['min'], value))
            
            # Round to the specified precision
            return round(value, range_info['precision'])
        return round(float(value), 2)
    except (ValueError, TypeError):
        if feature_name in FEATURE_RANGES:
            return FEATURE_RANGES[feature_name]['default']
        return 0.0

def analyze_medical_data(vital_stats, risk_score):
    """Use Groq to analyze medical data and provide heart-disease specific insights"""
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Check if API key is properly set
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                logger.error("Groq API key not configured")
                return get_default_analysis(vital_stats, risk_score)

            # Create Groq client
            client = Groq(api_key=api_key)

            # Get patient age and risk level
            age = vital_stats['Patient Demographics']['Age']
            risk_level = "High Risk" if risk_score > 0.5 else "Low Risk"

            # Prepare the prompt with patient details and vital statistics
            prompt = f"""As a cardiologist, analyze these patient details and vital statistics for heart disease risk factors and provide detailed insights:

Patient Details:
- Age: {age}
- Sex: {vital_stats['Patient Demographics']['Sex']}
- Height: {vital_stats['Patient Demographics']['Height']} cm
- Weight: {vital_stats['Patient Demographics']['Weight']} kg
- Pregnancy Status: {vital_stats['Patient Demographics']['Pregnancy Status']}
- Risk Score: {risk_score:.2f} ({risk_level})

Murmur Assessment:
- Presence: {'Present' if vital_stats['Cardiovascular Metrics']['Murmur Assessment']['Presence'] else 'Absent'}
- Location: {vital_stats['Cardiovascular Metrics']['Murmur Assessment']['Location']}
- Most Audible Location: {vital_stats['Cardiovascular Metrics']['Murmur Assessment']['Most Audible Location']}

Systolic Murmur Details:
- Timing: {vital_stats['Cardiovascular Metrics']['Murmur Assessment']['Systolic Details']['Timing']}
- Shape: {vital_stats['Cardiovascular Metrics']['Murmur Assessment']['Systolic Details']['Shape']}
- Grade: {vital_stats['Cardiovascular Metrics']['Murmur Assessment']['Systolic Details']['Grade']}
- Pitch: {vital_stats['Cardiovascular Metrics']['Murmur Assessment']['Systolic Details']['Pitch']}
- Quality: {vital_stats['Cardiovascular Metrics']['Murmur Assessment']['Systolic Details']['Quality']}

Diastolic Murmur Details:
- Timing: {vital_stats['Cardiovascular Metrics']['Murmur Assessment']['Diastolic Details']['Timing']}
- Shape: {vital_stats['Cardiovascular Metrics']['Murmur Assessment']['Diastolic Details']['Shape']}
- Grade: {vital_stats['Cardiovascular Metrics']['Murmur Assessment']['Diastolic Details']['Grade']}
- Pitch: {vital_stats['Cardiovascular Metrics']['Murmur Assessment']['Diastolic Details']['Pitch']}
- Quality: {vital_stats['Cardiovascular Metrics']['Murmur Assessment']['Diastolic Details']['Quality']}

Please provide a comprehensive analysis including:
1. Age-specific risk factors and considerations
2. Risk level interpretation based on age and overall health
3. Evidence-based lifestyle recommendations appropriate for the patient's age
4. Urgent warning signs (if any)
5. Positive health indicators
6. Age-appropriate follow-up recommendations
7. Potential diagnostic tests to consider based on age and risk level

Format the response as a structured JSON with these sections:
- risk_factors (including age-specific factors)
- cardiovascular_health
- lifestyle_recommendations (age-appropriate)
- warning_signs
- positive_indicators
- follow_up_recommendations (age-specific)
- diagnostic_tests (age-appropriate)
"""

            # Try with primary model first
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a cardiologist specializing in heart disease risk assessment and prevention. Provide detailed, evidence-based analysis and recommendations that are specifically tailored to the patient's age group. For pediatric patients, focus on growth and development considerations. For adults, focus on age-appropriate lifestyle modifications and monitoring."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model="qwen-qwq-32b",  # Using the correct model name
                    temperature=0.3,
                    max_tokens=2048
                )
            except Exception as e:
                logger.error(f"Error with Groq API: {str(e)}")
                return get_default_analysis(vital_stats, risk_score)

            try:
                analysis = json.loads(chat_completion.choices[0].message.content)
                return analysis
            except (json.JSONDecodeError, AttributeError, IndexError) as e:
                logger.error(f"Error parsing Groq response: {str(e)}")
                if attempt == max_retries - 1:
                    return get_default_analysis(vital_stats, risk_score)
                time.sleep(retry_delay)
                continue

        except Exception as e:
            logger.error(f"Error in Groq analysis (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                return get_default_analysis(vital_stats, risk_score)
            time.sleep(retry_delay)
            continue

    return get_default_analysis(vital_stats, risk_score)

def get_default_analysis(vital_stats, risk_score):
    """Provide a default analysis when the AI service is unavailable"""
    demographics = vital_stats["Patient Demographics"]
    murmur = vital_stats["Cardiovascular Metrics"]["Murmur Assessment"]
    age = demographics['Age']
    
    risk_factors = []
    warnings = []
    
    # Age-specific risk factors
    if age == 'Child':
        risk_factors.append("Pediatric patient - requires specialized cardiac evaluation")
        if murmur["Presence"]:
            risk_factors.append("Heart murmur in pediatric patient - requires careful monitoring of growth and development")
    elif age == 'Adolescent':
        risk_factors.append("Adolescent patient - requires age-appropriate cardiac evaluation")
        if murmur["Presence"]:
            risk_factors.append("Heart murmur in adolescent - requires monitoring of physical development and activity tolerance")
    else:
        risk_factors.append("Adult patient - standard cardiac evaluation protocol")
        if murmur["Presence"]:
            risk_factors.append("Heart murmur in adult patient - requires comprehensive cardiac workup")
    
    # Murmur analysis
    if murmur["Presence"]:
        risk_factors.append(f"Heart murmur detected at {murmur['Location']}")
        if murmur['Systolic Details']['Grade'] in ['III/VI', 'IV/VI', 'V/VI', 'VI/VI']:
            warnings.append("Significant heart murmur detected - requires immediate evaluation")
    
    # BMI calculation and analysis
    height_m = float(demographics['Height']) / 100  # convert cm to m
    weight_kg = float(demographics['Weight'])
    bmi = weight_kg / (height_m * height_m)
    if bmi > 30:
        risk_factors.append("Obesity (BMI > 30)")
    elif bmi > 25:
        risk_factors.append("Overweight (BMI 25-30)")

    # Age-specific recommendations
    if age == 'Child':
        lifestyle_recommendations = [
            "Maintain a balanced diet appropriate for growth and development",
            "Engage in age-appropriate physical activity (30-60 minutes daily)",
            "Regular monitoring of growth parameters and milestones",
            "Schedule regular pediatric cardiology follow-ups",
            "Ensure proper vaccination schedule",
            "Limit screen time and encourage active play",
            "Maintain regular sleep schedule appropriate for age"
        ]
        follow_up_recommendations = [
            "Schedule pediatric cardiology consultation",
            "Regular growth and development assessment",
            "Monitor for any changes in murmur characteristics",
            "Regular echocardiogram as recommended by pediatric cardiologist",
            "Annual comprehensive pediatric check-up",
            "Regular monitoring of physical activity tolerance"
        ]
        diagnostic_tests = [
            "Pediatric echocardiogram",
            "Growth and development assessment",
            "Basic metabolic panel",
            "Regular cardiac monitoring",
            "Developmental screening",
            "Physical activity tolerance test"
        ]
    elif age == 'Adolescent':
        lifestyle_recommendations = [
            "Maintain a balanced diet with focus on proper nutrition for growth",
            "Engage in regular physical activity (60 minutes daily)",
            "Monitor blood pressure regularly",
            "Maintain healthy weight",
            "Schedule regular cardiac check-ups",
            "Limit processed foods and sugary drinks",
            "Ensure adequate sleep (8-10 hours)",
            "Practice stress management techniques"
        ]
        follow_up_recommendations = [
            "Schedule adolescent cardiology consultation",
            "Regular cardiac monitoring",
            "Annual comprehensive health check-up",
            "Regular blood pressure monitoring",
            "Monitor physical development and activity tolerance",
            "Regular assessment of lifestyle habits"
        ]
        diagnostic_tests = [
            "Echocardiogram",
            "Electrocardiogram (ECG)",
            "Basic metabolic panel",
            "Lipid profile",
            "Physical activity tolerance test",
            "Blood pressure monitoring"
        ]
    else:
        lifestyle_recommendations = [
            "Maintain a heart-healthy diet (Mediterranean or DASH diet)",
            "Engage in regular physical activity (150 minutes weekly)",
            "Monitor blood pressure regularly",
            "Maintain healthy weight",
            "Schedule regular cardiac check-ups",
            "Limit alcohol consumption",
            "Quit smoking if applicable",
            "Manage stress through relaxation techniques"
        ]
        follow_up_recommendations = [
            "Schedule cardiology consultation",
            "Regular cardiac monitoring",
            "Annual comprehensive health check-up",
            "Regular blood pressure monitoring",
            "Regular lipid profile testing",
            "Diabetes screening if indicated"
        ]
        diagnostic_tests = [
            "Echocardiogram",
            "Electrocardiogram (ECG)",
            "Stress test if indicated",
            "Basic metabolic panel",
            "Lipid profile",
            "Coronary calcium score if indicated"
        ]

    return {
        "risk_factors": risk_factors if risk_factors else ["No immediate risk factors identified based on available data"],
        "cardiovascular_health": {
            "murmur_status": "Present" if murmur["Presence"] else "Absent",
            "murmur_location": murmur["Location"],
            "overall_assessment": "Requires attention" if risk_score > 0.5 or murmur["Presence"] else "Generally stable"
        },
        "lifestyle_recommendations": lifestyle_recommendations,
        "warning_signs": warnings if warnings else ["No immediate warning signs detected"],
        "positive_indicators": [
            "Regular monitoring of heart health",
            "Seeking medical guidance for assessment",
            "No heart murmur detected" if not murmur["Presence"] else None
        ],
        "follow_up_recommendations": follow_up_recommendations,
        "diagnostic_tests": diagnostic_tests
    }

# Load Model & Scaler
try:
    model = tf.keras.models.load_model("models/heart_disease_model.h5")
    scaler = joblib.load("models/scaler.pkl")
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    raise

@app.route('/predict', methods=['GET', 'POST'])
@limiter.limit("30 per minute")
def predict():
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            patient_id = data.get('patient_id')
        else:
            patient_id = request.form.get('patient_id')
    else:
        patient_id = request.args.get('patient_id')
        
    logger.info(f"Received prediction request for patient ID: {patient_id}")

    # Validate Input
    if not patient_id:
        return jsonify({"error": "Missing 'patient_id' in request"}), 400

    try:
        patient_id = str(int(patient_id))  # Convert safely to int and back to string
    except ValueError:
        return jsonify({"error": "Invalid Patient ID. Must be an integer!"}), 400

    try:
        # First, get patient demographics from training_data.csv
        training_data = pd.read_csv("data/training_data.csv", encoding='utf-8', sep='\t')  # Specify tab separator
        training_data.columns = training_data.columns.str.strip()
        
        # Print column names for debugging
        logger.info(f"Training data columns: {training_data.columns.tolist()}")
        
        # Ensure required columns exist
        required_columns = ['Patient ID', 'Murmur', 'Murmur locations', 'Most audible location',
                          'Systolic murmur timing', 'Systolic murmur shape', 'Systolic murmur grading',
                          'Systolic murmur pitch', 'Systolic murmur quality', 'Diastolic murmur timing',
                          'Diastolic murmur shape', 'Diastolic murmur grading', 'Diastolic murmur pitch',
                          'Diastolic murmur quality']
        
        missing_columns = [col for col in required_columns if col not in training_data.columns]
        if missing_columns:
            logger.error(f"Missing required columns in training data: {missing_columns}")
            return jsonify({"error": f"Invalid training data format: missing columns {missing_columns}"}), 500
        
        # Convert Patient ID to string and handle any potential whitespace
        training_data['Patient ID'] = training_data['Patient ID'].astype(str).str.strip()
        
        patient_demographics = training_data[training_data['Patient ID'] == patient_id]
        
        if patient_demographics.empty:
            logger.warning(f"Patient ID {patient_id} not found in training data")
            return jsonify({"error": f"Patient demographics not found for ID {patient_id}"}), 404
            
        patient_demographics = patient_demographics.iloc[0]
        
        # Log the raw murmur data for debugging
        logger.info(f"Raw murmur data for patient {patient_id}:")
        logger.info(f"Murmur: {patient_demographics['Murmur']}")
        logger.info(f"Murmur locations: {patient_demographics['Murmur locations']}")
        logger.info(f"Most audible location: {patient_demographics['Most audible location']}")
        logger.info(f"Systolic murmur timing: {patient_demographics['Systolic murmur timing']}")
        
        # Extract demographic details with proper error handling
        try:
            demographics = {
                "Age": str(patient_demographics['Age']),
                "Sex": str(patient_demographics['Sex']),
                "Height": float(patient_demographics['Height']) if pd.notna(patient_demographics['Height']) else 0.0,
                "Weight": float(patient_demographics['Weight']) if pd.notna(patient_demographics['Weight']) else 0.0,
                "Pregnancy Status": str(patient_demographics['Pregnancy status'])
            }
        except KeyError as e:
            logger.error(f"Missing required column in training data: {str(e)}")
            return jsonify({"error": f"Invalid training data format: missing {str(e)}"}), 500

        # Then, get features from processed_features.csv for prediction
        patient_data = DATA_DF[DATA_DF['patient_id'] == patient_id]
        
        if patient_data.empty:
            logger.warning(f"Patient ID {patient_id} not found in processed features dataset")
            return jsonify({"error": f"Patient features not found for ID {patient_id}"}), 404

        # Extract Features (Ensure 32 features)
        X = patient_data.iloc[:, 1:33].values  # Ensuring 32 features only
        logger.info(f"Features extracted for patient ID {patient_id}. Shape: {X.shape}")

        # Validate feature count
        if X.shape[1] != 32:
            logger.error(f"Invalid feature count: {X.shape[1]} for patient ID {patient_id}")
            return jsonify({"error": f"Data format error: Expected 32 features, but got {X.shape[1]}"}), 400

        # Normalize Features for prediction
        X_scaled = scaler.transform(X)
        
        # Reshape Correctly
        X_reshaped = X_scaled.reshape(-1, 32, 1)
        
        # Make Prediction
        prediction = model.predict(X_reshaped)[0][0]
        logger.info(f"Prediction made successfully for patient ID {patient_id}: {prediction}")

        # Determine risk level
        risk_level = "High Risk" if prediction > 0.5 else "Low Risk"

        # Process vital statistics with proper normalization
        vital_stats = {
            "Patient Demographics": demographics,
            "Cardiovascular Metrics": {
                "Murmur Assessment": {
                    "Presence": bool(X[0][7]),  # feat_7 indicates murmur presence
                    "Location": "Multiple" if X[0][8] > 0.5 else "Single",  # feat_8 indicates location complexity
                    "Most Audible Location": "Aortic" if X[0][9] > 0.5 else "Mitral",  # feat_9 indicates primary location
                    "Systolic Details": {
                        "Timing": "Early" if X[0][10] > 0.5 else "Late",  # feat_10 indicates timing
                        "Shape": "Crescendo" if X[0][11] > 0.5 else "Decrescendo",  # feat_11 indicates shape
                        "Grade": f"{int(X[0][12] * 6)}/VI",  # feat_12 scaled to grade I-VI
                        "Pitch": "High" if X[0][13] > 0.5 else "Low",  # feat_13 indicates pitch
                        "Quality": "Harsh" if X[0][14] > 0.5 else "Blowing"  # feat_14 indicates quality
                    },
                    "Diastolic Details": {
                        "Timing": "Early" if X[0][15] > 0.5 else "Late",  # feat_15 indicates timing
                        "Shape": "Crescendo" if X[0][16] > 0.5 else "Decrescendo",  # feat_16 indicates shape
                        "Grade": f"{int(X[0][17] * 6)}/VI",  # feat_17 scaled to grade I-VI
                        "Pitch": "High" if X[0][18] > 0.5 else "Low",  # feat_18 indicates pitch
                        "Quality": "Harsh" if X[0][19] > 0.5 else "Blowing"  # feat_19 indicates quality
                    }
                }
            }
        }
        
        # Log the murmur details for debugging
        logger.info(f"Murmur details for patient {patient_id}:")
        logger.info(f"Raw features: {X[0][7:20]}")  # Log the relevant features
        logger.info(f"Processed details: {vital_stats['Cardiovascular Metrics']['Murmur Assessment']}")

        # Get detailed analysis from Groq
        detailed_analysis = analyze_medical_data(vital_stats, prediction)

        # Create detailed response
        response = {
            "patient_id": patient_id,
            "risk_assessment": {
                "risk_level": risk_level,
                "risk_score": float(prediction),
                "interpretation": "High risk of heart disease - immediate medical consultation recommended." if prediction > 0.7
                                else "Moderate to high risk - schedule a cardiac evaluation soon." if prediction > 0.5
                                else "Low to moderate risk - maintain heart-healthy lifestyle." if prediction > 0.3
                                else "Low risk - continue current healthy practices."
            },
            "vital_statistics": vital_stats,
            "detailed_analysis": detailed_analysis,
            "analysis_timestamp": pd.Timestamp.now().isoformat()
        }

        logger.info(f"Sending response for patient ID {patient_id}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing request for patient ID {patient_id}: {str(e)}")
        return jsonify({"error": "Error processing request"}), 500

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG)

