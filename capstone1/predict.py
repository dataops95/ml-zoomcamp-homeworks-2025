import joblib
import numpy as np

def load_model():
    model = joblib.load('models/heart_disease_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

def predict(features):
    """
    features: dict with keys matching column names
    Returns: prediction (0 or 1) and probability
    """
    model, scaler = load_model()
    
    # Convert to array in correct order
    feature_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                     'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    features_array = np.array([[features[f] for f in feature_order]])
    
    # Scale and predict
    features_scaled = scaler.transform(features_array)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return {
        'prediction': int(prediction),
        'risk': 'High' if prediction == 1 else 'Low',
        'probability': float(probability[1])
    }
