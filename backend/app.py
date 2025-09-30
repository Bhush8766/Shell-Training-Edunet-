
# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import pickle, json

BASE = Path(__file__).parent
MODEL_PATH = BASE / "sms_spam_model.pkl"
VECTORIZER_PATH = BASE / "vectorizer.pkl"
METRICS_PATH = Path(__file__).parent / "metrics.json"

app = Flask(__name__)
CORS(app, origins=["https://shell-training-edunet-08.onrender.com"])

# load model & vectorizer
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# load metrics if present
metrics = {}
if METRICS_PATH.exists():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "SMS Spam Detection API"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        message = data.get("message", "")
        if not message:
            return jsonify({"error": "no message provided"}), 400

        x = vectorizer.transform([message])
        pred = model.predict(x)[0]

        # get probability for the predicted class (robust)
        confidence = 1.0
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x)[0]
            # model.classes_ aligns with proba columns
            class_index = list(model.classes_).index(pred)
            confidence = float(proba[class_index])

        return jsonify({"prediction": str(pred), "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/metrics", methods=["GET"])
def get_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        return jsonify(metrics)
    return jsonify({"error": "metrics not found"}), 404


if __name__ == "__main__":
    app.run(debug=True)
