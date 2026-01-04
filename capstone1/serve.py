from flask import Flask, request, jsonify
import predict

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict_heart_disease():
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                          'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Make prediction
        result = predict.predict(data)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696, debug=False)
