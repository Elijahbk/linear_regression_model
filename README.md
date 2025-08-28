# Soil Moisture Prediction API

This project builds and deploys a machine learning model to predict soil moisture based on environmental and soil features. The solution includes data analysis, model training, and a FastAPI-based web API for predictions.

## Features
- Data visualization and feature engineering
- Model comparison: Linear Regression, Decision Tree, Random Forest
- Automatic selection and saving of the best model
- FastAPI endpoint for real-time predictions
- Ready for deployment on Render or similar platforms

## Dataset
- The dataset should be named `data_core.csv` and placed in the project root.
- The target column for prediction is `Moisture`.

## Training the Model
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the training script:
   ```bash
   python model.py
   ```
   This will generate `best_model.pkl` and `scaler.pkl` for API use.

## Running the API Locally
1. Ensure `best_model.pkl` and `scaler.pkl` are present.
2. Start the API server:
   ```bash
   uvicorn api:app --reload
   ```
3. Open your browser at [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

## API Usage
- **POST** `/predict`
- Request body example:
  ```json
  {
    "soil_ph": 6.5,
    "temperature": 25.0,
    "rainfall": 120.0,
    "nutrient_level": 80.0
  }
  ```
- Response example:
  ```json
  {
    "predicted_moisture_level": 45.2
  }
  ```

## Deployment on Render
1. Push your code to a GitHub repository.
2. Create a new Web Service on [Render](https://render.com/):
   - Environment: Python 3
   - Start command: `uvicorn api:app --host 0.0.0.0 --port 10000`
3. Add all files (`api.py`, `model.py`, `requirements.txt`, `best_model.pkl`, `scaler.pkl`, etc.)
4. After deployment, access your API at `https://your-app-url.onrender.com/docs`

## License
MIT
