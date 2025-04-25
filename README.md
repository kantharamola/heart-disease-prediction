# Heart Disease Prediction Web Application

![Heart Disease Prediction](https://img.shields.io/badge/AI-Heart%20Disease%20Prediction-ff1654)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A sophisticated web application that leverages machine learning and AI to predict heart disease risk. The system analyzes patient data, including heart murmur assessments, to provide detailed medical insights and personalized recommendations.

## 🌟 Features

- **Advanced Risk Prediction**
  - Machine learning-based heart disease risk assessment
  - Real-time analysis of cardiovascular metrics
  - Comprehensive murmur assessment analysis

- **AI-Powered Analysis**
  - Integration with Groq AI for detailed medical insights
  - Age-specific risk factor analysis
  - Personalized health recommendations

- **Professional Interface**
  - Modern, responsive web design
  - Real-time data visualization
  - Interactive patient data input
  - Detailed results dashboard

- **Robust Security**
  - Rate limiting for API protection
  - Secure headers implementation
  - Input validation and sanitization
  - Error handling and logging

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kantharamola/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/MacOS
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create `.env.local` configuration:
   ```env
   GROQ_API_KEY=your_api_key_here
   FLASK_ENV=development
   ```

5. Run the application:
   ```bash
   python app.py
   ```

## 🌐 Deployment

The application is configured for deployment on Render.com:

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure build settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
4. Set environment variables:
   - `GROQ_API_KEY`: Your Groq API key
   - `FLASK_ENV`: `production`

## 📡 API Documentation

### Endpoints

#### Home Page
- **GET** `/`
- Returns the main application interface
- Response: HTML

#### Prediction Endpoint
- **POST** `/predict`
- Request body:
  ```json
  {
    "patient_id": "string"
  }
  ```
- Response:
  ```json
  {
    "patient_id": "string",
    "risk_assessment": {
      "risk_level": "string",
      "risk_score": "float",
      "interpretation": "string"
    },
    "vital_statistics": {
      "Patient_Demographics": {},
      "Cardiovascular_Metrics": {}
    },
    "detailed_analysis": {
      "risk_factors": [],
      "lifestyle_recommendations": []
    }
  }
  ```

## 🔒 Security Features

- **Rate Limiting**
  - 200 requests per day
  - 50 requests per hour

- **Security Headers**
  - Content Security Policy (CSP)
  - X-Content-Type-Options
  - X-Frame-Options
  - X-XSS-Protection
  - Strict-Transport-Security

- **Data Protection**
  - Input validation
  - Error handling
  - Secure error logging
  - Data sanitization

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

## 📱 Demo

Visit the live demo: [Heart Disease Prediction App](https://kantharamola.github.io/heart-disease-prediction/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

For support, email kantharamola@gmail.com or open an issue in the repository.

