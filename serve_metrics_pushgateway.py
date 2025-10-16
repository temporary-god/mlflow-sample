#!/usr/bin/env python3
# serve_with_metrics.py - A/B Testing focused metrics with Pushgateway
from flask import Flask, request, jsonify
import os
import time
import numpy as np
import pandas as pd
import mlflow.pyfunc
from prometheus_client import (
    Counter, Histogram, Gauge, 
    CollectorRegistry, pushadd_to_gateway,
    REGISTRY
)
from collections import deque
import json
import threading
import atexit

app = Flask(__name__)

MODEL_PATH = os.environ.get("MLFLOW_MODEL_PATH", "/opt/ml/model")
VARIANT = os.environ.get("VARIANT", "champion")
MODEL_NAME = os.environ.get("MODEL_NAME", "student_offer")
IMAGE_VERSION = os.environ.get("IMAGE_VERSION", "unknown")
PUSHGATEWAY_URL = os.environ.get("PUSHGATEWAY_URL", "http://10.0.11.179:9092")
PUSH_INTERVAL = int(os.environ.get("PUSH_INTERVAL", "30"))  # seconds

# Create registry
registry = CollectorRegistry()

# A/B Testing Metrics
prediction_accuracy = Gauge(
    "model_prediction_accuracy_live", 
    "Live prediction accuracy when feedback is available", 
    ["variant", "model", "version"], 
    registry=registry
)

prediction_confidence_avg = Gauge(
    "model_prediction_confidence_avg", 
    "Average prediction confidence score", 
    ["variant", "model", "version"], 
    registry=registry
)

data_drift_score = Gauge(
    "model_data_drift_score", 
    "Data drift score compared to training data", 
    ["variant", "model", "version"], 
    registry=registry
)

business_conversion_rate = Gauge(
    "model_business_conversion_rate", 
    "Business conversion rate (when outcome is known)", 
    ["variant", "model", "version"], 
    registry=registry
)

prediction_distribution = Counter(
    "model_prediction_distribution", 
    "Distribution of model predictions", 
    ["variant", "model", "version", "prediction_class"], 
    registry=registry
)

error_rate_by_confidence = Counter(
    "model_errors_by_confidence", 
    "Errors grouped by confidence buckets", 
    ["variant", "model", "version", "confidence_bucket"], 
    registry=registry
)

# Request metrics
request_count = Counter(
    "model_requests_total", 
    "Total requests", 
    ["variant", "model", "version", "status"], 
    registry=registry
)

response_time = Histogram(
    "model_response_time_seconds", 
    "Response time", 
    ["variant", "model", "version"], 
    registry=registry
)

# Service health metric
service_up = Gauge(
    "model_service_up",
    "Service health status (1=up, 0=down)",
    ["variant", "model", "version"],
    registry=registry
)

# In-memory storage for live accuracy calculation
recent_predictions = deque(maxlen=1000)
reference_data = None

# Background thread for pushing metrics
push_thread = None
stop_push_thread = threading.Event()

def push_metrics_to_gateway():
    """Background task to push metrics to Pushgateway"""
    while not stop_push_thread.is_set():
        try:
            # Set service as up
            service_up.labels(
                variant=VARIANT, 
                model=MODEL_NAME, 
                version=IMAGE_VERSION
            ).set(1)
            
            # Push all metrics to gateway
            job_name = f"{MODEL_NAME}_{VARIANT}"
            instance = f"{VARIANT}_{IMAGE_VERSION}"
            
            pushadd_to_gateway(
                gateway=PUSHGATEWAY_URL,
                job=job_name,
                registry=registry,
                grouping_key={
                    'variant': VARIANT,
                    'model': MODEL_NAME,
                    'version': IMAGE_VERSION,
                    'instance': instance
                }
            )
            
            print(f"[INFO] Pushed metrics to Pushgateway: {PUSHGATEWAY_URL} (job={job_name})")
            
        except Exception as e:
            print(f"[ERROR] Failed to push metrics to Pushgateway: {e}")
        
        # Wait for next push interval
        stop_push_thread.wait(PUSH_INTERVAL)

def start_push_thread():
    """Start the background thread for pushing metrics"""
    global push_thread
    push_thread = threading.Thread(target=push_metrics_to_gateway, daemon=True)
    push_thread.start()
    print(f"[INFO] Started metrics push thread (interval={PUSH_INTERVAL}s)")

def stop_push_thread_handler():
    """Stop the background thread gracefully"""
    print("[INFO] Stopping metrics push thread...")
    stop_push_thread.set()
    if push_thread:
        push_thread.join(timeout=5)
    
    # Final push before shutdown
    try:
        service_up.labels(
            variant=VARIANT, 
            model=MODEL_NAME, 
            version=IMAGE_VERSION
        ).set(0)  # Mark service as down
        
        job_name = f"{MODEL_NAME}_{VARIANT}"
        instance = f"{VARIANT}_{IMAGE_VERSION}"
        
        pushadd_to_gateway(
            gateway=PUSHGATEWAY_URL,
            job=job_name,
            registry=registry,
            grouping_key={
                'variant': VARIANT,
                'model': MODEL_NAME,
                'version': IMAGE_VERSION,
                'instance': instance
            }
        )
        print("[INFO] Pushed final metrics (service_up=0)")
    except Exception as e:
        print(f"[WARN] Failed to push final metrics: {e}")

# Register cleanup handler
atexit.register(stop_push_thread_handler)

# Load reference data for drift calculation
def load_reference_data():
    global reference_data
    try:
        ref_path = "/opt/ml/model/data/student_marks.csv"
        if os.path.exists(ref_path):
            reference_data = pd.read_csv(ref_path)
            print(f"[INFO] Loaded reference data: {len(reference_data)} rows")
        else:
            print(f"[WARN] Reference data not found at {ref_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load reference data: {e}")

def calculate_data_drift(current_data):
    """Calculate simple data drift score"""
    if reference_data is None or len(current_data) == 0:
        return 0.0
    
    try:
        current_df = pd.DataFrame(current_data)
        
        ref_numeric = reference_data.select_dtypes(include=[np.number]).columns
        curr_numeric = current_df.select_dtypes(include=[np.number]).columns
        common_cols = list(set(ref_numeric) & set(curr_numeric))
        
        if len(common_cols) == 0:
            return 0.0
        
        drift_scores = []
        for col in common_cols:
            ref_mean = reference_data[col].mean()
            ref_std = reference_data[col].std()
            curr_mean = current_df[col].mean()
            
            if ref_std > 0:
                drift = abs(curr_mean - ref_mean) / ref_std
                drift_scores.append(drift)
        
        return np.mean(drift_scores) * 100 if drift_scores else 0.0
        
    except Exception as e:
        print(f"[ERROR] Drift calculation failed: {e}")
        return 0.0

def update_live_metrics():
    """Update metrics based on accumulated data"""
    if len(recent_predictions) < 10:
        return
    
    # Calculate average confidence
    confidences = [p.get('confidence', 0.5) for p in recent_predictions]
    avg_confidence = np.mean(confidences)
    prediction_confidence_avg.labels(
        variant=VARIANT, 
        model=MODEL_NAME, 
        version=IMAGE_VERSION
    ).set(avg_confidence)
    
    # Calculate data drift from recent requests
    recent_inputs = [p['input_data'] for p in recent_predictions if 'input_data' in p]
    if recent_inputs:
        drift = calculate_data_drift(recent_inputs[-50:])
        data_drift_score.labels(
            variant=VARIANT, 
            model=MODEL_NAME, 
            version=IMAGE_VERSION
        ).set(drift)
    
    # Calculate live accuracy if we have feedback
    predictions_with_feedback = [p for p in recent_predictions if 'actual_outcome' in p]
    if len(predictions_with_feedback) > 0:
        correct = sum(1 for p in predictions_with_feedback 
                     if p['prediction'] == p['actual_outcome'])
        accuracy = correct / len(predictions_with_feedback)
        prediction_accuracy.labels(
            variant=VARIANT, 
            model=MODEL_NAME, 
            version=IMAGE_VERSION
        ).set(accuracy)
    
    # Calculate conversion rate (business metric)
    conversions = [p for p in recent_predictions if p.get('converted', False)]
    if len(recent_predictions) > 0:
        conversion_rate = len(conversions) / len(recent_predictions)
        business_conversion_rate.labels(
            variant=VARIANT, 
            model=MODEL_NAME, 
            version=IMAGE_VERSION
        ).set(conversion_rate)

# Load model
try:
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    load_reference_data()
    print(f"[INFO] Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Cannot load model: {e}")
    model = None

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy" if model else "unhealthy",
        "variant": VARIANT,
        "model": MODEL_NAME,
        "version": IMAGE_VERSION,
        "pushgateway": PUSHGATEWAY_URL
    })

@app.route("/invocations", methods=["POST"])
def invocations():
    if model is None:
        request_count.labels(
            variant=VARIANT, 
            model=MODEL_NAME, 
            version=IMAGE_VERSION,
            status="error"
        ).inc()
        return jsonify({"error": "Model not loaded"}), 500
        
    start_time = time.time()
    
    try:
        payload = request.get_json(force=True)
        
        # Make prediction
        predictions = model.predict(payload)
        
        # Calculate confidence
        confidence = calculate_prediction_confidence(predictions)
        
        # Store prediction for metrics
        pred_record = {
            'timestamp': time.time(),
            'prediction': predictions[0] if hasattr(predictions, '__len__') else predictions,
            'confidence': confidence,
            'input_data': payload[0] if isinstance(payload, list) else payload
        }
        recent_predictions.append(pred_record)
        
        # Update prediction distribution
        pred_class = str(predictions[0] if hasattr(predictions, '__len__') else predictions)
        prediction_distribution.labels(
            variant=VARIANT, 
            model=MODEL_NAME, 
            version=IMAGE_VERSION,
            prediction_class=pred_class
        ).inc()
        
        # Update confidence bucket
        confidence_bucket = get_confidence_bucket(confidence)
        error_rate_by_confidence.labels(
            variant=VARIANT,
            model=MODEL_NAME,
            version=IMAGE_VERSION,
            confidence_bucket=confidence_bucket
        ).inc()
        
        # Record response time
        response_time.labels(
            variant=VARIANT, 
            model=MODEL_NAME, 
            version=IMAGE_VERSION
        ).observe(time.time() - start_time)
        
        request_count.labels(
            variant=VARIANT, 
            model=MODEL_NAME, 
            version=IMAGE_VERSION,
            status="success"
        ).inc()
        
        # Update live metrics
        update_live_metrics()
        
        # Return prediction with metadata
        result = {
            "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else [predictions],
            "confidence": confidence,
            "variant": VARIANT,
            "model": MODEL_NAME,
            "version": IMAGE_VERSION,
            "prediction_id": len(recent_predictions)
        }
        
        return jsonify(result)
        
    except Exception as e:
        request_count.labels(
            variant=VARIANT, 
            model=MODEL_NAME, 
            version=IMAGE_VERSION,
            status="error"
        ).inc()
        return jsonify({"error": str(e)}), 500

@app.route("/feedback", methods=["POST"])
def feedback():
    """Endpoint to receive actual outcomes for accuracy calculation"""
    try:
        data = request.get_json()
        prediction_id = data.get('prediction_id')
        actual_outcome = data.get('actual_outcome')
        converted = data.get('converted', False)
        
        if prediction_id and prediction_id <= len(recent_predictions):
            idx = prediction_id - 1
            if idx >= 0:
                recent_predictions[idx]['actual_outcome'] = actual_outcome
                recent_predictions[idx]['converted'] = converted
                
                update_live_metrics()
                
                return jsonify({"status": "feedback_recorded"})
        
        return jsonify({"error": "Invalid prediction_id"}), 400
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def calculate_prediction_confidence(predictions):
    """Calculate confidence score based on prediction"""
    try:
        if hasattr(predictions, 'max'):
            return float(np.max(predictions))
        elif hasattr(predictions, '__len__') and len(predictions) > 0:
            return 0.85
        else:
            return 0.85
    except:
        return 0.85

def get_confidence_bucket(confidence):
    """Bucket confidence scores for error analysis"""
    if confidence >= 0.9:
        return "high"
    elif confidence >= 0.7:
        return "medium"
    else:
        return "low"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"[INFO] Starting {VARIANT} A/B testing model server on {host}:{port}")
    print(f"[INFO] Model: {MODEL_NAME}, Version: {IMAGE_VERSION}")
    print(f"[INFO] Pushgateway: {PUSHGATEWAY_URL}, Push interval: {PUSH_INTERVAL}s")
    
    # Start background thread for pushing metrics
    start_push_thread()
    
    # Run Flask app
    app.run(host=host, port=port)