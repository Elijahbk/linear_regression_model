import joblib
from model import scaler

joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as scaler.pkl")
