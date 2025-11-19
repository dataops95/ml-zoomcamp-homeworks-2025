# Import necessary libraries
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pickle

# Load the California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Create a DataFrame from the dataset
df = housing.frame
df.to_csv('data/data.csv', index=False)

# Check for missing values and fill them with median values
print(f"\n{df.isnull().sum()}")

# Fill NA values with median of each numeric column
medians = df.median(numeric_only=True)
df.fillna(medians, inplace=True)

# Verify there are no remaining missing values
print("\nMissing values after imputation:")
print(df.isnull().sum())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Standartization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def print_metrics_report(model_name, mae, rmse, r2):
    print(f"--- {model_name} Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared Error (R2): {r2:.4f}")
    print("-------------------------------------")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=10, random_state=42),
}

print(f"{'Model':<20} | {'MAE':<10} | {'RMSE':<10} | {'R^2':<10}")
print(f"{'-'*20} | {'-'*10} | {'-'*10} | {'-'*10}")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mae, rmse, r2 = calculate_metrics(y_test, y_pred)
    print(f"{name:<20} | {mae:<10.4f} | {rmse:<10.4f} | {r2:<10.4f}")

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred = lr_model.predict(X_test_scaled)
mae, rmse, r2 = calculate_metrics(y_test, y_pred)
print_metrics_report("LinearRegression", mae, rmse, r2)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)
mae, rmse, r2 = calculate_metrics(y_test, y_pred)
print_metrics_report("RandomForestRegressor", mae, rmse, r2)

# XGBoost Regressor Model
xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=10, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
y_pred = xgb_model.predict(X_test_scaled)
mae, rmse, r2 = calculate_metrics(y_test, y_pred)
print_metrics_report("XGBRegressor", mae, rmse, r2)

# The Best Model
# Train the XGBoost Regressor Model
model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# Model Storing
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))