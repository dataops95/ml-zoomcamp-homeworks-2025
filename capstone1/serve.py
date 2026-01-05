"""
Heart Disease Prediction - Flask API Server
This script serves the trained model via REST API
"""

from flask import Flask, request, jsonify
import predict
import logging
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize predictor
predictor = predict.HeartDiseasePredictor()

@app.before_first_request
def load_model():
    """Load model when the app starts"""
    logger.info("Starting Flask server...")
    if predictor.load_model():
        logger.info("✅ Model loaded successfully")
    else:
        logger.error("❌ Failed to load model")

@app.route('/')
def home():
    """Home page with API documentation"""
    return jsonify({
        'message': 'Heart Disease Prediction API',
        'version': '1.0.0',
        'endpoints': {
            'GET /': 'API documentation',
            'GET /health': 'Health check',
            'GET /info': 'Model information',
            'POST /predict': 'Make prediction for single patient',
            'POST /predict_batch': 'Make predictions for multiple patients'
        },
        'example_request': {
            'age': 45,
            'sex': 1,
            'cp': 2,
            'trestbps': 130,
            'chol': 230,
            'fbs': 0,
            'restecg': 1,
            'thalach': 150,
            'exang': 0,
            'oldpeak': 0.5,
            'slope': 2,
            'ca': 0,
            'thal': 2
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if predictor.is_loaded:
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'timestamp': datetime.now().isoformat()
        }), 200
    else:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route('/info', methods=['GET'])
def model_info():
    """Return model information"""
    if not predictor.is_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'model_name': predictor.metadata['model_name'],
        'model_type': predictor.metadata['model_type'],
        'training_date': predictor.metadata['training_date'],
        'n_features': predictor.metadata['n_features'],
        'feature_names': predictor.metadata['feature_names'],
        'metrics': predictor.metadata['metrics'],
        'dataset_info': {
            'training_samples': predictor.metadata['training_samples'],
            'test_samples': predictor.metadata['test_samples'],
            'total_samples': predictor.metadata['dataset_size']
        }
    })

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Make prediction for a single patient
    
    Expected JSON format:
    {
        "age": 45,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 230,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 0.5,
        "slope": 2,
        "ca": 0,
        "thal": 2
    }
    
    Returns:
    {
        "prediction": 0 or 1,
        "risk_level": "Low Risk" or "High Risk",
        "probability": 0.0 to 1.0,
        "confidence": 0.0 to 1.0
    }
    """
    try:
        # Check if model is loaded
        if not predictor.is_loaded:
            return jsonify({'error': 'Model not loaded'}), 503
        
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Required fields
        required_fields = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        # Check for missing fields
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields
            }), 400
        
        # Log request
        logger.info(f"Received prediction request: age={data.get('age')}, sex={data.get('sex')}")
        
        # Make prediction
        result = predictor.predict(data)
        
        # Log result
        logger.info(f"Prediction result: {result['risk_level']} (probability: {result['probability']:.2%})")
        
        return jsonify(result), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch_endpoint():
    """
    Make predictions for multiple patients
    
    Expected JSON format:
    {
        "patients": [
            {
                "age": 45,
                "sex": 1,
                ...
            },
            {
                "age": 60,
                "sex": 0,
                ...
            }
        ]
    }
    
    Returns:
    {
        "predictions": [
            {"prediction": 0, "risk_level": "Low Risk", ...},
            {"prediction": 1, "risk_level": "High Risk", ...}
        ]
    }
    """
    try:
        # Check if model is loaded
        if not predictor.is_loaded:
            return jsonify({'error': 'Model not loaded'}), 503
        
        # Get request data
        data = request.get_json()
        
        if not data or 'patients' not in data:
            return jsonify({'error': 'No patients data provided'}), 400
        
        patients = data['patients']
        
        if not isinstance(patients, list):
            return jsonify({'error': 'Patients must be a list'}), 400
        
        if len(patients) == 0:
            return jsonify({'error': 'Empty patients list'}), 400
        
        # Log request
        logger.info(f"Received batch prediction request for {len(patients)} patients")
        
        # Make predictions
        results = predictor.predict_batch(patients)
        
        # Log results
        high_risk_count = sum(1 for r in results if r.get('prediction') == 1)
        logger.info(f"Batch prediction complete: {high_risk_count}/{len(patients)} high risk")
        
        return jsonify({'predictions': results}), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("="*80)
    print("HEART DISEASE PREDICTION API")
    print("="*80)
    print("\nStarting Flask server...")
    print("API will be available at: http://localhost:9696")
    print("\nEndpoints:")
    print("  • GET  /          - API documentation")
    print("  • GET  /health    - Health check")
    print("  • GET  /info      - Model information")
    print("  • POST /predict   - Single prediction")
    print("  • POST /predict_batch - Batch predictions")
    print("\nExample curl command:")
    print("""
    curl -X POST http://localhost:9696/predict \\
      -H "Content-Type: application/json" \\
      -d '{
        "age": 45,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 230,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 0.5,
        "slope": 2,
        "ca": 0,
        "thal": 2
      }'
    """)
    print("="*80 + "\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=9696, debug=False)