#!/usr/bin/env python3
"""
train_and_log_model_with_path.py

Usage example:
python train_and_log_model_with_path.py \
  --reference_path ./data/student_marks.csv \
  --threshold 80 \
  --epochs 1 \
  --tracking_uri http://10.0.11.179:5000 \
  --experiment_name sixdee_experiments \
  --pushgateway_url http://10.0.11.179:9091
"""
import argparse
import os
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, ColSpec
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from typing import Any, Dict, Optional


class StudentOfferLabelModel(mlflow.pyfunc.PythonModel):
    """
    mlflow.pyfunc model object. Implements:
      - fit(): helper used during training
      - predict(): required for pyfunc serving
      - load_context(): called by mlflow when the model is loaded in a different environment.
    """

    def __init__(self, threshold: float = 80.0, epochs: int = 1):
        self.pipeline = None
        self.threshold = threshold
        self.epochs = max(1, int(epochs))
        # When the model is loaded by MLflow model serve, load_context will set this path
        self.reference_csv_path = None

    def fit(self, reference_data: Optional[pd.DataFrame] = None):
        """
        Fit the internal pipeline using reference_data DataFrame (or fallback to reading local path).
        Returns (accuracy, reference_df)
        """
        if reference_data is None:
            # fallback to reading file named 'student_marks.csv' in cwd
            if os.path.exists("student_marks.csv"):
                reference_data = pd.read_csv("student_marks.csv")
            else:
                raise FileNotFoundError(
                    "No reference_data provided and student_marks.csv not found in cwd."
                )

        data = reference_data.copy()
        # Create label column the same way your original code did
        data["placed"] = (data["marks"] > self.threshold).astype(int)
        X = data[["marks"]].astype(float)
        y = data["placed"].astype(int)

        self.pipeline = Pipeline(
            [("scaler", StandardScaler()), ("lr", LogisticRegression())]
        )

        for _ in range(self.epochs):
            self.pipeline.fit(X, y)

        preds = self.pipeline.predict(X)
        acc = accuracy_score(y, preds)
        return acc, data

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        """
        model_input is expected to be a DataFrame or convertible to one and contain a 'marks' column.
        Returns "Placed"/"Not Placed" strings.
        """
        if self.pipeline is None:
            raise RuntimeError("Model pipeline is not fitted. Call fit() before predict().")
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        preds = self.pipeline.predict(model_input[["marks"]].astype(float))
        return np.where(preds == 1, "Placed", "Not Placed").astype(str)

    def load_context(self, context):
        """
        Called by MLflow when model is loaded in a different environment (e.g., inside the served Docker image).
        mlflow passes a context.artifacts mapping: keys are artifact names used when logging the model,
        values are local paths where mlflow extracted them inside the container.
        """
        try:
            # We logged the artifact with key 'data/student_marks.csv' (path-key),
            # so look it up using that exact key.
            self.reference_csv_path = context.artifacts.get("data/student_marks.csv")
        except Exception:
            self.reference_csv_path = None

        # Optionally, preload reference_data or do other init here
        if self.reference_csv_path and os.path.exists(self.reference_csv_path):
            try:
                ref_df = pd.read_csv(self.reference_csv_path)
                # you could compute stats or warm caches here if desired
                # e.g., self.ref_stats = ref_df.describe()
            except Exception:
                pass


def _ensure_url_scheme(url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return "http://" + url


def push_metrics_to_prometheus(
    train_acc: float,
    drift_score: float,
    pushgateway_url: str,
    job_name: str,
    image_name: str,
    image_version: str,
    grouping_key: Optional[Dict[str, str]] = None,
):
    pushgateway_url = _ensure_url_scheme(pushgateway_url)
    registry = CollectorRegistry()

    # numeric metrics
    accuracy_metric = Gauge(
        "student_model_train_accuracy", "Training accuracy", registry=registry
    )
    drift_metric = Gauge(
        "student_model_drift_score", "Data drift score (percent)", registry=registry
    )

    accuracy_metric.set(float(train_acc))
    drift_metric.set(float(drift_score))

    # build/info metric (labels for name + version). value=1 just indicates existence.
    build_info = Gauge(
        "student_model_build_info",
        "Build info: image name and version (value=1 means present)",
        ["image_name", "image_version"],
        registry=registry,
    )
    # set label combination to 1
    build_info.labels(image_name="evidently-metrics-student-offer_label_test", image_version="64").set(1)

    try:
        push_to_gateway(
            pushgateway_url,
            job=job_name,
            registry=registry,
            grouping_key=grouping_key or {},
        )
        print(
            f"✅ Pushed metrics to Pushgateway at {pushgateway_url} (job={job_name}, grouping_key={grouping_key}, image={image_name}:{image_version})"
        )
    except Exception as e:
        print(f"⚠️ Failed to push metrics to Pushgateway at {pushgateway_url}: {e}")

def _recursive_find(obj: Any, target_key: str) -> Optional[Any]:
    """
    Recursively search dictionaries/lists for the first occurrence of target_key and return its value.
    """
    if isinstance(obj, dict):
        if target_key in obj:
            return obj[target_key]
        for v in obj.values():
            found = _recursive_find(v, target_key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _recursive_find(item, target_key)
            if found is not None:
                return found
    return None


def main(args):
    # Connect to mlflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # Validate reference path exists where given
    if not os.path.isfile(args.reference_path):
        raise FileNotFoundError(f"Reference CSV not found at: {args.reference_path}")

    # Read the reference data (this is the baseline used by Evidently)
    reference_data = pd.read_csv(args.reference_path)

    # Build and train model (we pass the DataFrame rather than hardcoding a name)
    model = StudentOfferLabelModel(threshold=args.threshold, epochs=args.epochs)

    # Start MLflow run
    if mlflow.active_run() is None:
        run_cm = mlflow.start_run(run_name="student_model_with_drift_check")
    else:
        run_cm = mlflow.start_run(run_name="student_model_with_drift_check", nested=True)

    with run_cm as run:
        print(f"RUN_ID: {run.info.run_id}")

        # Fit using provided reference_data (so we don't rely on a hardcoded file inside fit())
        acc, ref_df = model.fit(reference_data=reference_data)

        # Create signature for logging model
        signature = ModelSignature(
            inputs=Schema([ColSpec("double", "marks")]),
            outputs=Schema([ColSpec("string", "prediction")]),
        )

        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_metric("train_accuracy", acc)

        # Log the reference CSV as a run artifact (traceability) AND include it in model artifacts
        # so the model bundle contains it for serving and for the docker build path.
        try:
            mlflow.log_artifact(args.reference_path, artifact_path="data")
            print(f"✅ Logged reference CSV as run artifact: {args.reference_path}")
        except Exception as e:
            print(f"⚠️ Failed to log reference CSV as run artifact: {e}")

        # Prepare artifacts mapping for model: key -> path inside model bundle, value -> local path
        # NOTE: using a path-like key so it lands under /opt/ml/model/data/student_marks.csv in the image.
        artifacts = {"data/student_marks.csv": args.reference_path}

        # Log the mlflow.pyfunc model and include the CSV in the model bundle
        try:
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=model,
                signature=signature,
                artifacts=artifacts,  # bundle data/student_marks.csv into the model artifact
            )
            print("✅ Logged pyfunc model and included reference CSV as model artifact under data/")
        except Exception as e:
            print(f"⚠️ Failed to log model with artifacts: {e}")
            raise

        # Create synthetic current_data (you may replace with real incoming batch)
        np.random.seed(42)
        current_data = pd.DataFrame(
            {
                "student": [f"S{i+11}" for i in range(10)],
                "marks": np.random.normal(loc=82, scale=5, size=10).round().astype(int),
            }
        )
        current_data["placed"] = (current_data["marks"] > args.threshold).astype(int)

        # Run Evidently report using the in-memory reference_data you read above
        drift_score_percent = 0.0
        try:
            report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
            report.run(reference_data=ref_df, current_data=current_data)

            # Save and log report
            report_path = "drift_report.html"
            report.save_html(report_path)
            mlflow.log_artifact(report_path, artifact_path="reports")
            print(f"✅ Evidently report saved at {report_path} and logged to MLflow")

            # Extract drift metrics robustly
            report_dict = report.as_dict()
            # Prefer a normalized share if present; otherwise derive from counts
            num_drifted = _recursive_find(report_dict, "number_of_drifted_columns")
            total_cols = (
                _recursive_find(report_dict, "number_of_columns")
                or len(ref_df.columns)
                or None
            )
            drift_share = _recursive_find(report_dict, "drift_share")
            drift_score_field = _recursive_find(report_dict, "drift_score")

            drift_fraction = 0.0
            if drift_share is not None:
                drift_fraction = float(drift_share)
            elif num_drifted is not None and total_cols:
                drift_fraction = float(num_drifted) / float(total_cols)
            elif drift_score_field is not None:
                drift_fraction = float(drift_score_field)
            else:
                drift_fraction = 0.0

            # Use percent (0..100) for Prometheus metric
            drift_score_percent = float(drift_fraction) * 100.0
            print(f"Extracted drift fraction={drift_fraction}, percent={drift_score_percent}")

        except Exception as e:
            print(f"ERROR while running Evidently report or extracting drift: {e}")
            drift_score_percent = 0.0

        # Push metrics
        grouping = {
            "run_id": run.info.run_id,
            "experiment": args.experiment_name,
            "model": "student_offer",
        }
        push_metrics_to_prometheus(
            train_acc=acc,
            drift_score=drift_score_percent,
            pushgateway_url=args.pushgateway_url,
            job_name="student_model_monitoring",
            grouping_key=grouping,
        )

        # Print MLflow UI link for convenience
        print(
            f"🔗 View in MLflow UI: {args.tracking_uri}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--tracking_uri", type=str, default="http://10.0.11.179:5000")
    parser.add_argument("--experiment_name", type=str, default="sixdee_experiments")
    parser.add_argument(
        "--pushgateway_url", type=str, default="http://10.0.11.179:9091"
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        required=True,
        help="Local path to reference CSV that will be used as baseline and bundled into model artifacts.",
    )
    args = parser.parse_args()

    main(args)

