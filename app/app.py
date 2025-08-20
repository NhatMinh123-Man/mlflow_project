from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import json

app = Flask(__name__)

# Đường dẫn
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(CURRENT_DIR, "..", "model"))
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
BEST_ACC_PATH = os.path.join(MODEL_DIR, "accuracy.txt")
META_PATH = os.path.join(MODEL_DIR, "meta.json")

# Load model tốt nhất
model = joblib.load(BEST_MODEL_PATH)

# Đọc meta (đặc biệt là n_features)
if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    NUM_FEATURES = int(meta.get("n_features", 20))
    MODEL_META = meta
else:
    NUM_FEATURES = 20
    MODEL_META = {}


def get_fixed_accuracy():
    try:
        with open(BEST_ACC_PATH, "r") as f:
            s = f.read().strip()
            return float(s)
    except:
        return None


def get_confidence(X):
    try:
        proba = model.predict_proba(X).max()
        return float(proba)
    except Exception:
        return None


@app.route("/")
def home():
    return render_template("index.html", accuracy=get_fixed_accuracy(), meta=MODEL_META)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [float(v) for v in request.form.values()]
        if len(values) != NUM_FEATURES:
            raise ValueError(f"Expect {NUM_FEATURES} features, got {len(values)}")

        X = np.array(values).reshape(1, -1)
        pred = int(model.predict(X)[0])
        conf = get_confidence(X)

        return render_template(
            "index.html",
            prediction_text=f"Prediction: {pred}",
            form_values=values,
            accuracy=get_fixed_accuracy(),
            confidence=None if conf is None else round(conf, 3),
            meta=MODEL_META,
        )
    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {e}",
            accuracy=get_fixed_accuracy(),
            meta=MODEL_META,
        )


@app.route("/random_predict")
def random_predict():
    try:
        raw = np.random.uniform(-5, 5, NUM_FEATURES)
        rounded = np.round(raw, 1)
        X = rounded.reshape(1, -1)

        pred = int(model.predict(X)[0])
        conf = get_confidence(X)

        return render_template(
            "index.html",
            prediction_text=f"Random prediction: {pred}",
            random_values=rounded.tolist(),
            form_values=rounded.tolist(),  # điền sẵn vào form
            accuracy=get_fixed_accuracy(),
            confidence=None if conf is None else round(conf, 3),
            meta=MODEL_META,
        )
    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {e}",
            accuracy=get_fixed_accuracy(),
            meta=MODEL_META,
        )


# API JSON (Postman/curl)
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    X = np.array(data["features"]).reshape(1, -1)
    pred = int(model.predict(X)[0])
    conf = get_confidence(X)
    return jsonify(
        {
            "prediction": pred,
            "confidence": conf,
            "fixed_accuracy": get_fixed_accuracy(),
            "n_features": NUM_FEATURES,
            "meta": MODEL_META,
        }
    )


if __name__ == "__main__":
    # Chạy Flask
    app.run(debug=True)
