# Import necessary libraries
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pickle

# Load the California Housing dataset
df_original = fetch_california_housing()
housing = fetch_california_housing(as_frame=True)

# Create a DataFrame from the dataset
df_raw = housing.frame
df_raw.to_csv('data/data_raw.csv', index=False)

df_original = df_raw.rename(columns={
    "MedInc": "median_income",
    "HouseAge": "median_house_age",
    "AveRooms": "avg_rooms",
    "AveBedrms": "avg_bedrooms",
    "Population": "population",
    "AveOccup": "avg_occupants",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "MedHouseVal": "PRICE"
})

# Check for missing values
print(f"\n{df_original.isnull().sum()}")

# Fill NA values with median of each numeric column
medians = df_original.median(numeric_only=True)
df_original.fillna(medians, inplace=True)

# Verify there are no remaining missing values
print("\nMissing values after imputation:")
print(df_original.isnull().sum())

# Data pre-processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Target
df = df_original.drop(columns=['latitude', 'longitude'])
X = df_original.drop('PRICE', axis=1)
y = df_original['PRICE']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Standartization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
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


# XGBoost Regressor Model
model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
mae, rmse, r2 = calculate_metrics(y_test, y_pred)
print_metrics_report("XGBRegressor", mae, rmse, r2)

# Models Storing
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("Model and scaler have been saved successfully.")