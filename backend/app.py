# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load model and vectorizer
MODEL_PATH = Path(__file__).parent / "sms_spam_model.pkl"
VECTORIZER_PATH = Path(__file__).parent / "vectorizer.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# Home route
@app.route('/', methods=['GET'])
def home():
    return "SMS Spam Detection API is running!"

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        message = data.get("message")
        x = vectorizer.transform([message])
        pred = model.predict(x)[0]
        confidence = model.predict_proba(x).max() if hasattr(model, "predict_proba") else 1.0
        return jsonify({"prediction": pred, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
