# src/train.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

def train_model():
    data = pd.read_csv("data/dataset.csv")
    print(data.head()) 
    X = data[["feature"]]
    y = data["label"]

    model = LinearRegression()
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    print("Model trained and saved successfully.")
    return model

if __name__ == "__main__":
    train_model()
