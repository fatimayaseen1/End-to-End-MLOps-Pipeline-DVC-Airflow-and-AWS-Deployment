# tests/test_train_module.py
import os
from src.train import train_model

def test_train_model_runs():
    model = train_model()
    assert model is not None

def test_model_file_created():
    train_model()
    assert os.path.exists("models/model.pkl")
