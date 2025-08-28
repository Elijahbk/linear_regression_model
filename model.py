import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("data_core.csv")

# Check for non-numeric columns
dtypes = data.dtypes
non_numeric = dtypes[dtypes == 'object'].index.tolist()
if non_numeric:
    print(f"Non-numeric columns found: {non_numeric}")
    # Example: Convert categorical columns to numeric (if any)
    data = pd.get_dummies(data, columns=non_numeric, drop_first=True)

# Feature Engineering: Drop highly correlated features if correlation > 0.9
corr_matrix = data.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
if to_drop:
    print(f"Dropping highly correlated features: {to_drop}")
    data = data.drop(columns=to_drop)

# Update features and target after feature engineering
feature_cols = [col for col in data.columns if col != "Moisture"]
X = data[feature_cols]
y = data["Moisture"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained and saved as model.pkl")

# Data Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Distribution of features
for col in feature_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Scatter plot of target vs. each feature
for col in feature_cols:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=data[col], y=data["Moisture"])
    plt.title(f'{col} vs. Moisture')
    plt.xlabel(col)
    plt.ylabel('Moisture')
    plt.show()

# Train and evaluate models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}
results = {}
loss_curves = {}

for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    y_train_pred = mdl.predict(X_train)
    y_test_pred = mdl.predict(X_test)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    results[name] = {'train_mse': train_mse, 'test_mse': test_mse, 'train_r2': train_r2, 'test_r2': test_r2}
    # For loss curve (MSE per epoch, only for Linear Regression with gradient descent)
    if name == 'Linear Regression':
        # Simulate loss curve using SGDRegressor for demonstration
        from sklearn.linear_model import SGDRegressor
        sgd = SGDRegressor(max_iter=200, tol=1e-3, random_state=42)
        train_losses, test_losses = [], []
        for i in range(1, 201):
            sgd.partial_fit(X_train, y_train)
            train_losses.append(mean_squared_error(y_train, sgd.predict(X_train)))
            test_losses.append(mean_squared_error(y_test, sgd.predict(X_test)))
        loss_curves['train'] = train_losses
        loss_curves['test'] = test_losses

# Plot loss curve for Linear Regression (SGD)
plt.figure(figsize=(8, 5))
plt.plot(loss_curves['train'], label='Train Loss')
plt.plot(loss_curves['test'], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss Curve (SGDRegressor)')
plt.legend()
plt.show()

# Scatter plot: predictions vs actual (best model)
best_model_name = max(results, key=lambda k: results[k]['test_r2'])
best_model = models[best_model_name]
y_pred = best_model.predict(X_test)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Moisture Level')
plt.ylabel('Predicted Moisture Level')
plt.title(f'{best_model_name}: Actual vs Predicted')
plt.show()

print('Model comparison results:')
for name, res in results.items():
    print(f"{name}: Test R2 = {res['test_r2']:.3f}, Test MSE = {res['test_mse']:.3f}")

# Save the best model
joblib.dump(best_model, 'best_model.pkl')
print(f"Best model ({best_model_name}) saved as best_model.pkl")

# Function for prediction (to be used in API)
def predict_moisture(input_data):
    input_scaled = scaler.transform([input_data])
    return best_model.predict(input_scaled)[0]
