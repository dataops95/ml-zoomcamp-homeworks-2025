"""
Heart Disease Prediction - Model Training Script
Dataset: heart.csv (1,025 patients, 14 columns)
This script trains multiple models and saves the best one for deployment.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

# Configuration
DATA_PATH = 'data/heart.csv'
MODEL_DIR = 'models'
RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the heart disease dataset
    
    Expected format:
    - 1,025 rows
    - 14 columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, 
                  exang, oldpeak, slope, ca, thal, target
    """
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    print(f"✅ Dataset loaded: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Validate expected columns
    expected_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    if not all(col in df.columns for col in expected_cols):
        missing = [col for col in expected_cols if col not in df.columns]
        raise ValueError(f"❌ Missing expected columns: {missing}")
    
    # Target variable analysis
    print(f"\n📊 Target Variable Analysis:")
    print(f"   Unique values: {sorted(df['target'].unique())}")
    print(f"   Value counts:\n{df['target'].value_counts().sort_index()}")
    
    # Convert multi-class to binary if needed
    if df['target'].nunique() > 2:
        print(f"\n⚠️  Multi-class target detected. Converting to binary:")
        print(f"   0 → 0 (No disease)")
        print(f"   1-{df['target'].max()} → 1 (Disease present)")
        df['target'] = (df['target'] > 0).astype(int)
    
    target_dist = df['target'].value_counts().sort_index()
    print(f"\n✅ Final target distribution:")
    print(f"   No disease (0): {target_dist[0]} ({target_dist[0]/len(df)*100:.1f}%)")
    print(f"   Disease (1):    {target_dist[1]} ({target_dist[1]/len(df)*100:.1f}%)")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠️  Missing values detected:")
        print(missing[missing > 0])
        print("   Removing rows with missing values...")
        df = df.dropna()
        print(f"   After removal: {df.shape}")
    else:
        print(f"\n✅ No missing values detected")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"\n⚠️  Found {duplicates} duplicate rows. Removing...")
        df = df.drop_duplicates()
        print(f"   After removal: {df.shape}")
    else:
        print(f"✅ No duplicates detected")
    
    # Data quality checks
    print(f"\n📊 Data Quality Checks:")
    
    # Check for invalid values
    issues = []
    
    # Cholesterol: should be > 0 and < 600
    if 'chol' in df.columns:
        invalid_chol = ((df['chol'] == 0) | (df['chol'] > 600)).sum()
        if invalid_chol > 0:
            issues.append(f"cholesterol: {invalid_chol} invalid values")
    
    # Blood pressure: should be between 80 and 220
    if 'trestbps' in df.columns:
        invalid_bp = ((df['trestbps'] < 80) | (df['trestbps'] > 220)).sum()
        if invalid_bp > 0:
            issues.append(f"blood pressure: {invalid_bp} invalid values")
    
    # Age: should be between 20 and 100
    if 'age' in df.columns:
        invalid_age = ((df['age'] < 20) | (df['age'] > 100)).sum()
        if invalid_age > 0:
            issues.append(f"age: {invalid_age} invalid values")
    
    if issues:
        print(f"   ⚠️  Issues found: {', '.join(issues)}")
        print("   Note: Keeping outliers as they may represent real medical cases")
    else:
        print(f"   ✅ All values within expected ranges")
    
    print(f"\n✅ Final dataset shape: {df.shape}")
    print(f"   Samples: {len(df)}")
    print(f"   Features: {df.shape[1] - 1}")
    
    return df

def prepare_data(df):
    """
    Split data into features and target, then train/test sets
    """
    print("\n" + "="*80)
    print("PREPARING DATA FOR TRAINING")
    print("="*80)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Total samples: {len(df)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Feature names: {list(X.columns)}")
    
    # Class distribution
    class_counts = y.value_counts().sort_index()
    class_ratio = class_counts.min() / class_counts.max()
    
    print(f"\n📊 Class Distribution:")
    print(f"   Class 0 (No disease): {class_counts[0]} ({class_counts[0]/len(y)*100:.1f}%)")
    print(f"   Class 1 (Disease):    {class_counts[1]} ({class_counts[1]/len(y)*100:.1f}%)")
    print(f"   Balance ratio: {class_ratio:.2f}")
    
    if class_ratio < 0.5:
        print(f"   ⚠️  Imbalanced dataset - consider using class weights")
    else:
        print(f"   ✅ Dataset is reasonably balanced")
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"\n📊 Train/Test Split (test_size={TEST_SIZE}):")
    print(f"   Training samples: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"   Test samples:     {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")
    print(f"   Train distribution: {dict(y_train.value_counts().sort_index())}")
    print(f"   Test distribution:  {dict(y_test.value_counts().sort_index())}")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Standardize features using StandardScaler
    """
    print("\n" + "="*80)
    print("FEATURE SCALING")
    print("="*80)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("\n✅ Features scaled using StandardScaler")
    print(f"   Scaled mean (should be ~0.0): {X_train_scaled.mean().mean():.6f}")
    print(f"   Scaled std (should be ~1.0):  {X_train_scaled.std().mean():.6f}")
    
    return X_train_scaled, X_test_scaled, scaler

def train_models(X_train, y_train, X_test, y_test):
    """
    Train multiple classification models
    """
    print("\n" + "="*80)
    print("TRAINING MULTIPLE MODELS")
    print("="*80)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, eval_metric='logloss')
    }
    
    results = {}
    
    print("\n🚀 Training models...\n")
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
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
        
        print(f"  ✅ Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(f"     F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
        print(f"     CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})\n")
    
    return results

def tune_best_model(X_train, y_train, X_test, y_test, model_name, base_model):
    """
    Hyperparameter tuning using GridSearchCV
    """
    print("\n" + "="*80)
    print(f"HYPERPARAMETER TUNING - {model_name}")
    print("="*80)
    
    # Define parameter grids
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
    elif 'Gradient Boosting' in model_name:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1]
        }
    elif 'Logistic' in model_name:
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        }
    else:
        print("⚠️  No tuning grid for this model. Using default parameters.")
        return base_model
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\n🔍 Testing {total_combinations} parameter combinations with 5-fold CV...")
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    print("🚀 Starting Grid Search...")
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Grid Search Complete!")
    print(f"\n🏆 Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    
    print(f"\n📊 Performance:")
    print(f"   Best CV Score: {grid_search.best_score_:.4f}")
    
    # Test performance
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print(f"   Test ROC-AUC:  {test_roc_auc:.4f}")
    
    return best_model

def save_artifacts(model, scaler, feature_names, metadata):
    """
    Save all model artifacts
    """
    print("\n" + "="*80)
    print("SAVING MODEL ARTIFACTS")
    print("="*80)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    artifacts = {
        'model': (f'{MODEL_DIR}/heart_disease_model.pkl', model),
        'scaler': (f'{MODEL_DIR}/scaler.pkl', scaler),
        'features': (f'{MODEL_DIR}/feature_names.pkl', feature_names),
        'metadata': (f'{MODEL_DIR}/model_metadata.pkl', metadata)
    }
    
    for name, (path, obj) in artifacts.items():
        joblib.dump(obj, path)
        print(f"✅ {name.capitalize()} saved: {path}")
    
    print(f"\n📦 All artifacts saved in '{MODEL_DIR}/' directory")

def main():
    """
    Main training pipeline
    """
    start_time = datetime.now()
    
    print("="*80)
    print("HEART DISEASE PREDICTION - MODEL TRAINING")
    print("="*80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {DATA_PATH}")
    print("="*80)
    
    try:
        # 1. Load data
        df = load_and_preprocess_data(DATA_PATH)
        
        # 2. Prepare train/test split
        X_train, X_test, y_train, y_test = prepare_data(df)
        
        # 3. Scale features
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        # 4. Train multiple models
        results = train_models(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # 5. Select best model
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        best_metrics = results[best_model_name]
        
        print("\n" + "="*80)
        print(f"🏆 BEST MODEL: {best_model_name}")
        print("="*80)
        print(f"   Accuracy:  {best_metrics['accuracy']:.4f}")
        print(f"   Precision: {best_metrics['precision']:.4f}")
        print(f"   Recall:    {best_metrics['recall']:.4f}")
        print(f"   F1-Score:  {best_metrics['f1_score']:.4f}")
        print(f"   ROC-AUC:   {best_metrics['roc_auc']:.4f}")
        print(f"   CV Score:  {best_metrics['cv_mean']:.4f} (±{best_metrics['cv_std']:.4f})")
        
        # 6. Tune best model
        tuned_model = tune_best_model(
            X_train_scaled, y_train, X_test_scaled, y_test,
            best_model_name, results[best_model_name]['model']
        )
        
        # 7. Final evaluation
        y_pred_final = tuned_model.predict(X_test_scaled)
        y_pred_proba_final = tuned_model.predict_proba(X_test_scaled)[:, 1]
        
        final_metrics = {
            'accuracy': accuracy_score(y_test, y_pred_final),
            'precision': precision_score(y_test, y_pred_final, zero_division=0),
            'recall': recall_score(y_test, y_pred_final, zero_division=0),
            'f1_score': f1_score(y_test, y_pred_final, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba_final)
        }
        
        # 8. Prepare metadata
        metadata = {
            'model_name': f'{best_model_name} (Tuned)',
            'model_type': type(tuned_model).__name__,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_path': DATA_PATH,
            'dataset_size': len(df),
            'n_features': X_train.shape[1],
            'feature_names': list(X_train.columns),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE,
            'metrics': final_metrics,
            'cv_score': best_metrics['cv_mean']
        }
        
        # 9. Save artifacts
        save_artifacts(tuned_model, scaler, list(X_train.columns), metadata)
        
        # 10. Final summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"\n📊 Final Model: {metadata['model_name']}")
        print(f"📊 Dataset: {metadata['dataset_size']} samples, {metadata['n_features']} features")
        print(f"\n📊 Test Set Performance:")
        print(f"   Accuracy:  {final_metrics['accuracy']:.4f}")
        print(f"   Precision: {final_metrics['precision']:.4f}")
        print(f"   Recall:    {final_metrics['recall']:.4f}")
        print(f"   F1-Score:  {final_metrics['f1_score']:.4f}")
        print(f"   ROC-AUC:   {final_metrics['roc_auc']:.4f}")
        print(f"\n⏱️  Training Duration: {duration:.1f} seconds")
        print(f"📅 End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n✅ Model is ready for deployment!")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())