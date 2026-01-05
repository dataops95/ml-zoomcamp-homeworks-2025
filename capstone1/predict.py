"""
Heart Disease Prediction - Prediction Module
This module loads the trained model and makes predictions on new data.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Union, List

# Model paths
MODEL_PATH = 'models/heart_disease_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
FEATURE_NAMES_PATH = 'models/feature_names.pkl'
METADATA_PATH = 'models/model_metadata.pkl'

class HeartDiseasePredictor:
    """
    Predictor class for heart disease risk assessment
    """
    
    def __init__(self):
        """Initialize predictor by loading model artifacts"""
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        self.is_loaded = False
    
    def load_model(self):
        """Load model and preprocessing artifacts"""
        try:
            print("Loading model artifacts...")
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.feature_names = joblib.load(FEATURE_NAMES_PATH)
            self.metadata = joblib.load(METADATA_PATH)
            self.is_loaded = True
            print(f"✅ Model loaded: {self.metadata['model_name']}")
            print(f"   Trained on: {self.metadata['training_date']}")
            # print(f"   Test Accuracy: {self.metadata['metrics']['accuracy']:.4f}")
            print(f"   Test Accuracy: {self.metadata['test_accuracy']:.4f}")
            return True
        except FileNotFoundError as e:
            print(f"❌ Error: Model files not found. Please run train.py first.")
            print(f"   Missing file: {e.filename}")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def validate_input(self, features: Dict) -> bool:
        """
        Validate input features
        
        Args:
            features: Dictionary with feature values
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Check if all required features are present
        missing_features = set(self.feature_names) - set(features.keys())
        if missing_features:
            print(f"❌ Missing features: {missing_features}")
            return False
        
        # Check for extra features
        extra_features = set(features.keys()) - set(self.feature_names)
        if extra_features:
            print(f"⚠️  Extra features (will be ignored): {extra_features}")
        
        return True
    
    def preprocess_input(self, features: Dict) -> np.ndarray:
        """
        Preprocess input features for prediction
        
        Args:
            features: Dictionary with feature values
            
        Returns:
            Preprocessed feature array
        """
        # Create DataFrame with features in correct order
        feature_dict = {feature: [features[feature]] for feature in self.feature_names}
        df = pd.DataFrame(feature_dict)
        
        # Scale features
        scaled_features = self.scaler.transform(df)

        # Convert back to DataFrame to preserve feature names
        scaled_df = pd.DataFrame(scaled_features, columns=self. feature_names)
        
        return scaled_df
    
    def predict(self, features: Dict) -> Dict:
        """
        Make prediction on patient data
        
        Args:
            features: Dictionary containing patient features:
                - age: Age in years (int)
                - sex: Sex (1 = male, 0 = female) (int)
                - cp: Chest pain type (0-3) (int)
                - trestbps: Resting blood pressure (mm Hg) (int)
                - chol: Serum cholesterol (mg/dl) (int)
                - fbs: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) (int)
                - restecg: Resting ECG results (0-2) (int)
                - thalach: Maximum heart rate achieved (int)
                - exang: Exercise induced angina (1 = yes, 0 = no) (int)
                - oldpeak: ST depression (float)
                - slope: Slope of peak exercise ST segment (0-2) (int)
                - ca: Number of major vessels (0-3) (int)
                - thal: Thalassemia (1-3) (int)
        
        Returns:
            Dictionary with prediction results:
                - prediction: Binary prediction (0 or 1)
                - risk_level: Risk level ('Low Risk' or 'High Risk')
                - probability: Probability of disease (0-1)
                - confidence: Prediction confidence (0-1)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Validate input
        if not self.validate_input(features):
            raise ValueError("Invalid input features")
        
        # Preprocess
        scaled_features = self.preprocess_input(features)
        
        # Make prediction
        prediction = self.model.predict(scaled_features)[0]
        probabilities = self.model.predict_proba(scaled_features)[0]
        
        # Extract results
        probability_disease = float(probabilities[1])
        confidence = float(max(probabilities))
        
        result = {
            'prediction': int(prediction),
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
            'probability': probability_disease,
            'confidence': confidence
        }
        
        return result
    
    def predict_batch(self, features_list: List[Dict]) -> List[Dict]:
        """
        Make predictions on multiple patients
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of prediction results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = []
        for features in features_list:
            try:
                result = self.predict(features)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        
        return results
    
    def explain_prediction(self, features: Dict) -> Dict:
        """
        Provide explanation for the prediction (feature importance)
        
        Args:
            features: Dictionary with feature values
            
        Returns:
            Dictionary with prediction and feature contributions
        """
        result = self.predict(features)
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importances))
            
            # Sort by importance
            sorted_importance = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            result['feature_importance'] = dict(sorted_importance[:5])  # Top 5
        
        return result

# Example usage functions
def predict_single_patient(patient_data: Dict) -> Dict:
    """
    Convenience function for single patient prediction
    
    Args:
        patient_data: Dictionary with patient features
        
    Returns:
        Prediction results
    """
    predictor = HeartDiseasePredictor()
    
    if not predictor.load_model():
        return {'error': 'Failed to load model'}
    
    try:
        result = predictor.predict(patient_data)
        return result
    except Exception as e:
        return {'error': str(e)}

def main():
    """
    Demo: Make predictions on example patients
    """
    print("="*80)
    print("HEART DISEASE PREDICTION - DEMO")
    print("="*80)
    
    # Initialize predictor
    predictor = HeartDiseasePredictor()
    
    if not predictor.load_model():
        return
    
    # Example patients
    example_patients = [
        {
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
        },
        {
            'age': 60,
            'sex': 0,
            'cp': 3,
            'trestbps': 150,
            'chol': 280,
            'fbs': 1,
            'restecg': 0,
            'thalach': 130,
            'exang': 1,
            'oldpeak': 2.3,
            'slope': 1,
            'ca': 2,
            'thal': 3
        },
        {
            'age': 35,
            'sex': 1,
            'cp': 0,
            'trestbps': 120,
            'chol': 200,
            'fbs': 0,
            'restecg': 1,
            'thalach': 170,
            'exang': 0,
            'oldpeak': 0.0,
            'slope': 2,
            'ca': 0,
            'thal': 1
        }
    ]
    
    # Make predictions
    print("\n" + "="*80)
    print("PREDICTIONS")
    print("="*80)
    
    for i, patient in enumerate(example_patients, 1):
        print(f"\n📋 Patient {i}:")
        print(f"   Age: {patient['age']}, Sex: {'Male' if patient['sex'] == 1 else 'Female'}")
        print(f"   Blood Pressure: {patient['trestbps']}, Cholesterol: {patient['chol']}")
        
        result = predictor.predict(patient)
        
        print(f"\n🔮 Prediction Results:")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Probability of Disease: {result['probability']:.2%}")
        print(f"   Confidence: {result['confidence']:.2%}")
        
        if result['prediction'] == 1:
            print(f"   ⚠️  RECOMMENDATION: Further cardiac evaluation recommended")
        else:
            print(f"   ✅ RECOMMENDATION: Continue regular health monitoring")
        
        print("-"*80)
    
    print("\n✅ Demo complete!")

if __name__ == '__main__':
    main()