# Heart Disease Prediction Web Application

This is a web application that predicts heart disease risk using machine learning. The application provides detailed analysis and recommendations based on patient data.

## Features

- Heart disease risk prediction
- Detailed medical analysis using Groq AI
- Real-time vital statistics validation
- Interactive web interface
- Secure API endpoints with rate limiting

## Requirements

- Python 3.9+
- Dependencies listed in requirements.txt
- Groq API key for advanced analysis

## Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env.local` file with your configuration:
   ```
   GROQ_API_KEY=your_api_key_here
   FLASK_ENV=development
   ```
5. Run the application:
   ```bash
   python app.py
   ```

## Deployment

The application is configured for deployment on Render.com:

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set the following:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
4. Add environment variables:
   - GROQ_API_KEY
   - FLASK_ENV=production

## Security

- Rate limiting implemented
- Security headers configured
- Input validation
- Secure error handling

## API Documentation

### Endpoints

- GET `/`: Home page
- POST `/predict`: Prediction endpoint
  - Request body: `{ "patient_id": "string" }`
  - Response: JSON with prediction results and analysis

## License

MIT License

