#!/usr/bin/env python3
# serve_with_metrics.py
from flask import Flask, request, jsonify
import os
import time
import mlflow.pyfunc
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
from prometheus_client import multiprocess
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from prometheus_client import make_wsgi_app

app = Flask(__name__)

MODEL_PATH = os.environ.get("MLFLOW_MODEL_PATH", "/opt/ml/model")
VARIANT = os.environ.get("VARIANT", "champion")
IMAGE_NAME = os.environ.get("IMAGE_NAME", "unknown")
IMAGE_VERSION = os.environ.get("IMAGE_VERSION", "unknown")
MODEL_NAME = os.environ.get("MODEL_NAME", "student_offer")

# Create a per-process registry (simple, not multiprocess)
registry = CollectorRegistry()

# Metrics — keep labels minimal to avoid cardinality explosion
req_counter = Counter("model_requests_total", "Total requests to model", ["variant", "model"], registry=registry)
req_latency = Histogram("model_request_latency_seconds", "Request latency seconds", ["variant", "model"], registry=registry)
last_pred_count = Gauge("model_last_prediction_count", "Last prediction label counts", ["variant", "model", "label"], registry=registry)

# Load model once
try:
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    print(f"[INFO] Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Cannot load model at {MODEL_PATH}: {e}")
    model = None

@app.route("/invocations", methods=["POST"])
def invocations():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    start = time.time()
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON payload"}), 400

    # Accept either list-of-dicts or object convertible to pandas DataFrame
    try:
        preds = model.predict(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    latency = time.time() - start
    req_counter.labels(variant=VARIANT, model=MODEL_NAME).inc()
    req_latency.labels(variant=VARIANT, model=MODEL_NAME).observe(latency)

    # record simple label counts if preds are string-like
    try:
        from collections import Counter as C
        if hasattr(preds, "tolist"):
            items = preds.tolist()
        else:
            items = list(preds)
        counts = C(items)
        # set gauge for each observed label
        for label, cnt in counts.items():
            last_pred_count.labels(variant=VARIANT, model=MODEL_NAME, label=str(label)).set(cnt)
    except Exception:
        pass

    return jsonify(items)

@app.route("/metrics")
def metrics():
    data = generate_latest(registry)
    return data, 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    app.run(host=host, port=port)
