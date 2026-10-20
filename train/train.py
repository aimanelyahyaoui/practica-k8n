import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import r2_score

MODEL_PATH = os.environ.get("MODEL_PATH", "/data/model.pkl")

np.random.seed(2)
x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100) / x

mask = x > 0.1
x, y = x[mask], y[mask]

split = int(len(x) * 0.8)
train_x, test_x = x[:split].reshape(-1, 1), x[split:].reshape(-1, 1)
train_y, test_y = y[:split], y[split:]

model = Pipeline([
    ("inverse", FunctionTransformer(np.reciprocal)),   # x -> 1/x
    ("linreg", LinearRegression()),
])
model.fit(train_x, train_y)

print(f"R2 train: {r2_score(train_y, model.predict(train_x)):.3f}")
print(f"R2 test:  {r2_score(test_y, model.predict(test_x)):.3f}")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"Modelo guardado en {MODEL_PATH}")
