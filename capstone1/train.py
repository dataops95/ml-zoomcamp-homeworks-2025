# -*- coding: utf-8 -*-
"""
Heart Disease Prediction - Training Pipeline
Dataset: heart.csv
Purpose: Model training for production pipeline (No EDA)
"""

import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    average_precision_score
)

import xgboost as xgb
import joblib
import os

# Settings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

print("=" * 80)
print("HEART DISEASE PREDICTION - TRAINING PIPELINE")
print("=" * 80)

# ============================================================================
# Load Dataset
# ============================================================================
print("\n[1/7] Loading dataset...")
df = pd.read_csv('data/heart.csv')
print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# Remove duplicates if any
if df.duplicated().sum() > 0:
    df = df.drop_duplicates()
    print(f"✅ Duplicates removed. New shape: {df.shape}")

# ============================================================================
# Train/Test Split
# ============================================================================
print("\n[2/7] Splitting data...")
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"✅ Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"✅ Testing set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")

# ============================================================================
# Feature Scaling
# ============================================================================
print("\n[3/7] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
print("✅ Features scaled using StandardScaler")

# ============================================================================
# Model Training - Multiple Algorithms
# ============================================================================
print("\n[4/7] Training models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB()
}

results = {}

for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else np.nan
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    print(f"  {name:<25s}: Accuracy={accuracy:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")

print("✅ All models trained successfully!")

# ============================================================================
# Hyperparameter Tuning - Random Forest
# ============================================================================
print("\n[5/7] Hyperparameter tuning for Random Forest...")

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

grid_search_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

grid_search_rf.fit(X_train_scaled, y_train)

best_rf_model = grid_search_rf.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test_scaled)
y_pred_proba_best_rf = best_rf_model.predict_proba(X_test_scaled)[:, 1]
best_rf_accuracy = accuracy_score(y_test, y_pred_best_rf)
best_rf_f1 = f1_score(y_test, y_pred_best_rf)
best_rf_roc_auc = roc_auc_score(y_test, y_pred_proba_best_rf)

print(f"✅ RF Tuned: Accuracy={best_rf_accuracy:.4f}, F1={best_rf_f1:.4f}, ROC-AUC={best_rf_roc_auc:.4f}")

# ============================================================================
# Hyperparameter Tuning - XGBoost
# ============================================================================
print("\n[6/7] Hyperparameter tuning for XGBoost...")

param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

grid_search_xgb = GridSearchCV(
    xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
    param_grid_xgb,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

grid_search_xgb.fit(X_train_scaled, y_train)

best_xgb_model = grid_search_xgb.best_estimator_
y_pred_best_xgb = best_xgb_model.predict(X_test_scaled)
y_pred_proba_best_xgb = best_xgb_model.predict_proba(X_test_scaled)[:, 1]
best_xgb_accuracy = accuracy_score(y_test, y_pred_best_xgb)
best_xgb_f1 = f1_score(y_test, y_pred_best_xgb)
best_xgb_roc_auc = roc_auc_score(y_test, y_pred_proba_best_xgb)

print(f"✅ XGB Tuned: Accuracy={best_xgb_accuracy:.4f}, F1={best_xgb_f1:.4f}, ROC-AUC={best_xgb_roc_auc:.4f}")

# ============================================================================
# Final Model Selection
# ============================================================================
print("\n[7/7] Selecting best model...")

# Compare all models including tuned versions
all_models = {
    **{name: results[name] for name in results},
    'RF (Tuned)': {
        'model': best_rf_model,
        'accuracy': best_rf_accuracy,
        'f1_score': best_rf_f1,
        'roc_auc': best_rf_roc_auc
    },
    'XGBoost (Tuned)': {
        'model': best_xgb_model,
        'accuracy': best_xgb_accuracy,
        'f1_score': best_xgb_f1,
        'roc_auc': best_xgb_roc_auc
    }
}

# Select best model based on accuracy
best_model_name = max(all_models.keys(), key=lambda k: all_models[k]['accuracy'])
best_model = all_models[best_model_name]['model']
best_accuracy = all_models[best_model_name]['accuracy']
best_f1 = all_models[best_model_name]['f1_score']
best_roc_auc = all_models[best_model_name]['roc_auc']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.4f}")
print(f"   F1-Score: {best_f1:.4f}")
print(f"   ROC-AUC: {best_roc_auc:.4f}")

# ============================================================================
# Save Model Artifacts
# ============================================================================
print("\n" + "=" * 80)
print("SAVING MODEL ARTIFACTS")
print("=" * 80)

os.makedirs('models', exist_ok=True)

# Save the best model
model_filename = 'models/heart_disease_model.pkl'
joblib.dump(best_model, model_filename)
print(f"✅ Model saved: {model_filename}")

# Save the scaler
scaler_filename = 'models/scaler.pkl'
joblib.dump(scaler, scaler_filename)
print(f"✅ Scaler saved: {scaler_filename}")

# Save feature names
feature_names_filename = 'models/feature_names.pkl'
joblib.dump(list(X.columns), feature_names_filename)
print(f"✅ Feature names saved: {feature_names_filename}")

# Save model metadata
metadata = {
    'model_name': best_model_name,
    'model_type': type(best_model).__name__,
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'test_accuracy': float(best_accuracy),
    'test_f1_score': float(best_f1),
    'test_roc_auc': float(best_roc_auc),
    'n_features': X.shape[1],
    'feature_names': list(X.columns),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'dataset_size': len(df)
}

metadata_filename = 'models/model_metadata.pkl'
joblib.dump(metadata, metadata_filename)
print(f"✅ Model metadata saved: {metadata_filename}")

# Verify saved model
print("\n🔍 Verifying saved model...")
loaded_model = joblib.load(model_filename)
loaded_scaler = joblib.load(scaler_filename)
test_prediction = loaded_model.predict(loaded_scaler.transform(X_test.iloc[:1]))
print(f"✅ Model verification successful!")

print("\n" + "=" * 80)
print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\n📦 Saved files:")
print(f"  • {model_filename}")
print(f"  • {scaler_filename}")
print(f"  • {feature_names_filename}")
print(f"  • {metadata_filename}")
print(f"\n🚀 Model is ready for production deployment!")
