
import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field


# Project paths
#BASE_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent.parent

# Initialize FastAPI
app = FastAPI(
    title="House Price Prediction API",
    description="Predict California house prices using ML",
    version="1.0"
)


# Load trained model
model = joblib.load(
    BASE_DIR / "models" / "house_price_model.joblib"
)


# Input schema with validation
class HouseFeatures(BaseModel):

    MedInc: float = Field(gt=0)
    HouseAge: float = Field(gt=0)
    AveRooms: float = Field(gt=0)
    AveBedrms: float = Field(gt=0)
    Population: float = Field(gt=0)
    AveOccup: float = Field(gt=0)

    Latitude: float = Field(ge=32, le=42)
    Longitude: float = Field(ge=-125, le=-114)


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running!"
    }


# Prediction endpoint
@app.post("/predict")
def predict_price(data: HouseFeatures):

    input_data = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_data)[0]

    return {
        "predicted_price": round(float(prediction * 100000), 2),
        "currency": "USD"
    }
