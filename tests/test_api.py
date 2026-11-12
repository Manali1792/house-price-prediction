from fastapi.testclient import TestClient
from src.main import app
client = TestClient(app)

# Test 1: Home endpoint
def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "House Price Prediction API is running!"

# Test 2: Valid prediction
def test_predict():
    payload = {
        "MedInc": 5.0,
        "HouseAge": 25,
        "AveRooms": 6.0,
        "AveBedrms": 1.2,
        "Population": 1000,
        "AveOccup": 3.0,
        "Latitude": 34.05,
        "Longitude": -118.25
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "predicted_price" in response.json()
    assert response.json()["currency"] == "USD"

# Test 3: Invalid MedInc
def test_invalid_medincome():
    payload = {
        "MedInc": -5,
        "HouseAge": 25,
        "AveRooms": 6.0,
        "AveBedrms": 1.2,
        "Population": 1000,
        "AveOccup": 3.0,
        "Latitude": 34.05,
        "Longitude": -118.25
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

# Test 4: Invalid Latitude
def test_invalid_latitude():
    payload = {
        "MedInc": 5.0,
        "HouseAge": 25,
        "AveRooms": 6.0,
        "AveBedrms": 1.2,
        "Population": 1000,
        "AveOccup": 3.0,
        "Latitude": 50,
        "Longitude": -118.25
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
