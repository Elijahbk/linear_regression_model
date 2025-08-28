from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

# Load scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("best_model.pkl")

# Define feature names (update as needed)
FEATURES = ["soil_ph", "temperature", "rainfall", "nutrient_level"]

class MoistureInput(BaseModel):
    soil_ph: float = Field(..., ge=0, le=14, description="Soil pH (0-14)")
    temperature: float = Field(..., ge=-50, le=60, description="Temperature in Celsius")
    rainfall: float = Field(..., ge=0, le=1000, description="Rainfall in mm")
    nutrient_level: float = Field(..., ge=0, le=100, description="Nutrient level (0-100 scale)")

app = FastAPI(title="Soil Moisture Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(input: MoistureInput):
    data = np.array([[input.soil_ph, input.temperature, input.rainfall, input.nutrient_level]])
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)[0]
    return {"predicted_moisture_level": pred}
