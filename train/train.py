import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import os

np.random.seed(42)
X = np.random.rand(100, 1) * 10 + 1
y = 5 * (1 / X) + 2 + np.random.randn(100, 1) * 0.1

X_transformed = 1 / X

model = LinearRegression()
model.fit(X_transformed, y)

os.makedirs('/data', exist_ok=True)

with open('/data/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Modelo entrenado y guardado correctamente en /data/model.pkl")
