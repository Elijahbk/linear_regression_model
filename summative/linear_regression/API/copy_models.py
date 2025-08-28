# Copy your trained model files to the API directory for deployment
import shutil

shutil.copy('../../../../best_model.pkl', './best_model.pkl')
shutil.copy('../../../../scaler.pkl', './scaler.pkl')
