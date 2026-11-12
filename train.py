import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.data_loader import load_data
from src.preprocessing import create_preprocessor

# Load dataset
df = load_data()

# Separate features and target
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# Linear Regression Pipeline
linear_pipeline = Pipeline([
    ("preprocessing", create_preprocessor(scale_features=True)),
    ("model", LinearRegression())
])

# Random Forest Pipeline
rf_pipeline = Pipeline([
    ("preprocessing", create_preprocessor(scale_features=False)),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

models = {
    "Linear Regression": linear_pipeline,
    "Random Forest": rf_pipeline
}

results = []

for name, pipeline in models.items():

    print(f"\nTraining {name}...")

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })


# Display comparison
results_df = pd.DataFrame(results)

print("\n--- Model Comparison ---")
print(results_df.to_string(index=False))


import os
import joblib

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the trained Random Forest pipeline
joblib.dump(
    rf_pipeline,
    "models/house_price_model.joblib"
)

print("\nRandom Forest pipeline saved successfully!")
