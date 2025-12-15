import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load dataset
data = pd.read_csv("data/dataset.csv") 
X = data[["feature"]]
y = data["label"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/model.pkl")

print("Model trained and saved successfully.")
