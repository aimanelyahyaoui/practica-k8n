import os
import pickle
from flask import Flask, jsonify, request

MODEL_PATH = os.environ.get("MODEL_PATH", "/data/model.pkl")
app = Flask(__name__)
_model = None

def get_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model

@app.route("/")
def index():
    return """<h1>Model Server</h1>
    <p>Regresión lineal inversa (y = a·(1/x) + b): minutos antes de comprar -> dinero gastado.</p>
    <ul>
      <li><code>GET /model?x=3</code> devuelve la predicción en JSON</li>
      <li><code>GET /health</code> liveness</li>
      <li><code>GET /ready</code> readiness (comprueba que el modelo existe)</li>
    </ul>"""

@app.route("/model")
def predict():
    try:
        x = float(request.args["x"])
    except (KeyError, ValueError):
        return jsonify(error="Parámetro 'x' numérico requerido"), 400
    if x <= 0:
        return jsonify(error="x debe ser > 0"), 400
    try:
        return jsonify(x=x, prediction=float(get_model().predict([[x]])[0]))
    except FileNotFoundError:
        return jsonify(error="Modelo no disponible"), 503

@app.route("/health")
def health():
    return jsonify(status="ok")

@app.route("/ready")
def ready():
    if os.path.exists(MODEL_PATH):
        return jsonify(status="ready")
    return jsonify(status="model missing"), 503

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
