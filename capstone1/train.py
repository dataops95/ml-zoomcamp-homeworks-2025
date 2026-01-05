"""
Heart Disease Prediction - Model Training Script
This script trains multiple models and saves the best one for deployment.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

# Configuration
DATA_PATH = 'data/heart_disease.csv'
MODEL_DIR = 'models'
RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the heart disease dataset
    """
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    # Column mapping
    column_mapping = {
        'Age': 'age',
        'Sex': 'sex',
        'Chest pain type': 'cp',
        'BP': 'trestbps',
        'Cholesterol': 'chol',
        'FBS over 120': 'fbs',
        'EKG results': 'restecg',
        'Max HR': 'thalach',
        'Exercise angina': 'exang',
        'ST depression': 'oldpeak',
        'Slope of ST': 'slope',
        'Number of vessels fluro': 'ca',
        'Thallium': 'thal',
        'Heart Disease': 'target'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    print(f"✅ Dataset loaded: {df.shape}")
    
    # Encode target variable if it's a string
    if df['target'].dtype == 'object':
        label_encoder = LabelEncoder()
        df['target'] = label_encoder.fit_transform(df['target'])
        print(f"✅ Target variable encoded: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
        
        # Save label encoder
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(label_encoder, f'{MODEL_DIR}/label_encoder.pkl')
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"⚠️  Found {missing} missing values. Removing...")
        df = df.dropna()
    
    # Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"⚠️  Found {duplicates} duplicate rows. Removing...")
        df = df.drop_duplicates()
    
    print(f"✅ Final dataset shape: {df.shape}")
    
    return df

def prepare_data(df):
    """
    Split data into features and target, then train/test sets
    """
    print("\nPreparing data...")
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"✅ Training set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")
    print(f"   Class distribution (train): {dict(y_train.value_counts().sort_index())}")
    print(f"   Class distribution (test): {dict(y_test.value_counts().sort_index())}")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Standardize features using StandardScaler
    """
    print("\nScaling features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("✅ Features scaled successfully")
    
    return X_train_scaled, X_test_scaled, scaler

def train_models(X_train, y_train, X_test, y_test):
    """
    Train multiple models and compare performance
    """
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, eval_metric='logloss')
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"  ROC-AUC: {roc_auc:.4f}")
    
    return results

def tune_best_model(X_train, y_train, X_test, y_test, model_name, base_model):
    """
    Hyperparameter tuning for the best model
    """
    print(f"\n" + "="*80)
    print(f"HYPERPARAMETER TUNING - {model_name}")
    print("="*80)
    
    if 'Random Forest' in model_name:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif 'XGBoost' in model_name:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 1.0]
        }
    else:
        return base_model
    
    print(f"Testing {np.prod([len(v) for v in param_grid.values()])} combinations...")
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    
    print(f"\n📊 Best CV Score: {grid_search.best_score_:.4f}")
    
    # Test on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"📊 Test Set Accuracy: {test_accuracy:.4f}")
    
    return best_model

def save_model_artifacts(model, scaler, feature_names, metadata):
    """
    Save all model artifacts for deployment
    """
    print("\n" + "="*80)
    print("SAVING MODEL ARTIFACTS")
    print("="*80)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save model
    model_path = f'{MODEL_DIR}/heart_disease_model.pkl'
    joblib.dump(model, model_path)
    print(f"✅ Model saved: {model_path}")
    
    # Save scaler
    scaler_path = f'{MODEL_DIR}/scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved: {scaler_path}")
    
    # Save feature names
    features_path = f'{MODEL_DIR}/feature_names.pkl'
    joblib.dump(feature_names, features_path)
    print(f"✅ Feature names saved: {features_path}")
    
    # Save metadata
    metadata_path = f'{MODEL_DIR}/model_metadata.pkl'
    joblib.dump(metadata, metadata_path)
    print(f"✅ Metadata saved: {metadata_path}")
    
    print(f"\n📦 All artifacts saved in '{MODEL_DIR}/' directory")

def main():
    """
    Main training pipeline
    """
    print("="*80)
    print("HEART DISEASE PREDICTION - MODEL TRAINING")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load and preprocess data
    df = load_and_preprocess_data(DATA_PATH)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train multiple models
    results = train_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    
    print(f"\n🏆 BEST MODEL (before tuning): {best_model_name}")
    print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"   F1-Score: {results[best_model_name]['f1_score']:.4f}")
    print(f"   ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
    
    # Hyperparameter tuning
    tuned_model = tune_best_model(
        X_train_scaled, y_train, X_test_scaled, y_test,
        best_model_name, best_model
    )
    
    # Calculate final metrics
    y_pred_final = tuned_model.predict(X_test_scaled)
    y_pred_proba_final = tuned_model.predict_proba(X_test_scaled)[:, 1]
    
    final_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_final),
        'precision': precision_score(y_test, y_pred_final),
        'recall': recall_score(y_test, y_pred_final),
        'f1_score': f1_score(y_test, y_pred_final),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_final)
    }
    
    # Prepare metadata
    metadata = {
        'model_name': best_model_name + ' (Tuned)',
        'model_type': type(tuned_model).__name__,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_size': len(df),
        'n_features': X_train.shape[1],
        'feature_names': list(X_train.columns),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
        'metrics': final_metrics
    }
    
    # Save everything
    save_model_artifacts(tuned_model, scaler, list(X_train.columns), metadata)
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE - FINAL MODEL PERFORMANCE")
    print("="*80)
    print(f"\n📊 Model: {metadata['model_name']}")
    print(f"📊 Test Set Performance:")
    print(f"   Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"   Precision: {final_metrics['precision']:.4f}")
    print(f"   Recall:    {final_metrics['recall']:.4f}")
    print(f"   F1-Score:  {final_metrics['f1_score']:.4f}")
    print(f"   ROC-AUC:   {final_metrics['roc_auc']:.4f}")
    
    print("\n✅ Model is ready for deployment!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == '__main__':
    main()