import joblib
import pandas as pd
from src.data_loader import load_data

# Load saved model
model = joblib.load("models/house_price_model.joblib")

# Get one unseen test record
df = load_data()
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

sample = X.iloc[[100]]
actual = y.iloc[100]

# Predict
prediction = model.predict(sample)

print("Input data:")
print(sample)

print("\nPredicted House Value:", prediction[0] * 100000)
print("Actual House Value:", actual * 100000)
