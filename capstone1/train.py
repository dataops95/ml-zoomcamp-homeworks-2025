#!/usr/bin/env python
# coding: utf-8

"""
Heart Disease Prediction Model Training Script
This script trains the final model based on the best configuration from notebook.ipynb
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# File paths
DATA_PATH = 'data/heart.csv'
MODEL_OUTPUT_PATH = 'models/best_model.pkl'

def load_data(path):
    """Load the heart disease dataset"""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"Data shape: {df.shape}")
    return df

def preprocess_data(df):
    """Preprocess the data"""
    print("\nPreprocessing data...")
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=RANDOM_STATE):
    """Split data into train and test sets"""
    print(f"\nSplitting data (test_size={test_size}, random_state={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and compare performance"""
    print("\n" + "="*60)
    print("TRAINING MULTIPLE MODELS")
    print("="*60)
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            C=1.0,
            solver='lbfgs'
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            learning_rate=0.1,
            max_depth=5,
            eval_metric='logloss',
            use_label_encoder=False
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
    
    return results

def select_best_model(results, metric='roc_auc'):
    """Select the best model based on specified metric"""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        name: {
            'Accuracy': stats['accuracy'],
            'Precision': stats['precision'],
            'Recall': stats['recall'],
            'F1 Score': stats['f1'],
            'ROC-AUC': stats['roc_auc']
        }
        for name, stats in results.items()
    }).T
    
    print("\n" + comparison_df.to_string())
    
    # Select best model
    best_model_name = max(results.items(), key=lambda x: x[1][metric])[0]
    best_model = results[best_model_name]['model']
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"Selected based on: {metric.upper()} = {results[best_model_name][metric]:.4f}")
    print(f"{'='*60}")
    
    return best_model, best_model_name

def save_model(model, path=MODEL_OUTPUT_PATH):
    """Save the trained model to disk"""
    print(f"\nSaving model to {path}...")
    
    # Create directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved successfully!")

def evaluate_final_model(model, X_test, y_test):
    """Provide detailed evaluation of the final model"""
    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION")
    print("="*60)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nTrue Negatives:  {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives:  {cm[1,1]}")
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        print("\nTop 10 Most Important Features:")
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        print(feature_importance.to_string(index=False))

def main():
    """Main training pipeline"""
    print("="*60)
    print("HEART DISEASE PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Load data
    df = load_data(DATA_PATH)
    
    # Preprocess
    X, y = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train multiple models
    results = train_models(X_train, y_train, X_test, y_test)
    
    # Select best model (you can change metric to 'accuracy', 'f1', etc.)
    best_model, best_model_name = select_best_model(results, metric='roc_auc')
    
    # Detailed evaluation
    evaluate_final_model(best_model, X_test, y_test)
    
    # Save model
    save_model(best_model)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nBest Model: {best_model_name}")
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")
    print(f"\nYou can now run the prediction service with: python serve.py")

if __name__ == "__main__":
    main()
