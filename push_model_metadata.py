#!/usr/bin/env python3
"""
ops/push_model_metadata.py

Env variables expected:
  MODEL_NAME (e.g. evidently-metrics-student-offer)
  VERSION (e.g. 64)
  VARIANT (champion|challenger) - default: champion
  RUN_ID (optional) - mlflow run id to fetch logged metrics from
  PUSHGATEWAY_URL - default: http://10.0.11.179:9091
  MLFLOW_TRACKING_URI - default: http://10.0.11.179:5000
  JOB_NAME - Pushgateway job name, default: student_model_build_info
"""
import os
import sys
from mlflow.tracking import MlflowClient
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

MODEL_NAME = os.environ.get("MODEL_NAME")
VERSION = os.environ.get("VERSION")
VARIANT = os.environ.get("VARIANT", "champion")
RUN_ID = os.environ.get("RUN_ID")
PUSHGATEWAY = os.environ.get("PUSHGATEWAY_URL", "http://10.0.11.179:9091")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://10.0.11.179:5000")
JOB_NAME = os.environ.get("JOB_NAME", "student_model_build_info")

if not MODEL_NAME or not VERSION:
    print("ERROR: MODEL_NAME and VERSION environment variables are required", file=sys.stderr)
    sys.exit(2)

train_acc = None
drift_score = 0.0
experiment_id = None

# Attempt to get metrics from MLflow if run_id provided
if RUN_ID:
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        run = client.get_run(RUN_ID)
        experiment_id = run.info.experiment_id
        train_acc = run.data.metrics.get("train_accuracy")
        drift_score = float(run.data.metrics.get("drift_score_percent", 0.0))
    except Exception as e:
        print(f"[WARN] Failed to fetch run {RUN_ID} from MLflow: {e}", file=sys.stderr)
        train_acc = None
        drift_score = 0.0
else:
    print("[INFO] RUN_ID not provided — pushing with default/unknown numeric metrics", file=sys.stderr)

# Build registry & metrics
registry = CollectorRegistry()
g_acc = Gauge("student_model_train_accuracy", "Training accuracy", registry=registry)
g_drift = Gauge("student_model_drift_score", "Data drift score (percent)", registry=registry)
g_info = Gauge(
    "student_model_build_info",
    "Build info: image name and version (value=1 means present)",
    ["image_name", "image_version", "variant", "model"],
    registry=registry,
)

# set numeric metrics (fallback safe defaults)
try:
    g_acc.set(float(train_acc) if train_acc is not None else 0.0)
except Exception:
    g_acc.set(0.0)

g_drift.set(float(drift_score or 0.0))

g_info.labels(image_name=MODEL_NAME, image_version=VERSION, variant=VARIANT, model=MODEL_NAME).set(1)

grouping = {"variant": VARIANT, "image_name": MODEL_NAME, "image_version": VERSION, "model": MODEL_NAME}
if RUN_ID:
    grouping["run_id"] = RUN_ID

try:
    push_to_gateway(PUSHGATEWAY, job=JOB_NAME, registry=registry, grouping_key=grouping)
    print(f"✅ Pushed build metrics for {MODEL_NAME}:{VERSION} variant={VARIANT} to {PUSHGATEWAY}")
except Exception as e:
    print(f"⚠️ Failed to push to Pushgateway {PUSHGATEWAY}: {e}", file=sys.stderr)
    sys.exit(3)
