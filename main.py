from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# Load trained model
model = joblib.load("model.pkl")

# Define API input schema
class Features(BaseModel):
    soil_ph: float
    temperature: float
    rainfall: float
    nutrient_level: float

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Moisture Level Prediction API is running"}

@app.post("/predict")
def predict(data: Features):
    features = np.array([[data.soil_ph, data.temperature, data.rainfall, data.nutrient_level]])
    prediction = model.predict(features)[0]
    return {"predicted_moisture_level": float(prediction)}
