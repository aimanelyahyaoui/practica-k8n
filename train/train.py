import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

model_path = os.environ.get("MODEL_PATH", "/data/model.pkl")
os.makedirs