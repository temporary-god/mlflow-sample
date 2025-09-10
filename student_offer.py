#!/usr/bin/env python3
import argparse
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
import time
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


def _ensure_url_scheme(url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return "http://" + url


def push_metrics_to_prometheus(
    train_acc: float,
    drift_score: float,
    pushgateway_url: str,
    job_name: str,
    grouping_key: Optional[Dict[str, str]] = None,
):
    """
    Create a fresh registry, set gauges, and push to Pushgateway.
    This is safe to call repeatedly (e.g., per request).
    """
    pushgateway_url = _ensure_url_scheme(pushgateway_url)
    registry = CollectorRegistry()
    # Define gauges; names must match whatever you expect in Grafana/Prometheus
    accuracy_metric = Gauge("student_model_train_accuracy", "Training accuracy", registry=registry)
    drift_metric = Gauge("student_model_drift_score", "Data drift score", registry=registry)

    # Set values
    accuracy_metric.set(float(train_acc))
    drift_metric.set(float(drift_score))

    try:
        push_to_gateway(pushgateway_url, job=job_name, registry=registry, grouping_key=grouping_key or {})
        print(f"✅ Pushed metrics to Prometheus Pushgateway at {pushgateway_url} (job={job_name}, grouping_key={grouping_key})")
    except Exception as e:
        # don't raise — inference/training should continue even if metrics push fails
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


class StudentOfferLabelModel(mlflow.pyfunc.PythonModel):
    """
    MLflow PythonModel that:
      - fits a simple pipeline in fit()
      - stores train_acc and drift_score on the instance
      - pushes metrics to Pushgateway on every predict() invocation if pushgateway_url is configured
    """

    def __init__(
        self,
        threshold: float = 80.0,
        epochs: int = 1,
        pushgateway_url: Optional[str] = None,
        job_name: str = "student_model_monitoring",
        grouping_key: Optional[Dict[str, str]] = None,
    ):
        # model internals
        self.pipeline = None
        self.threshold = threshold
        self.epochs = max(1, int(epochs))

        # pushgateway config (None during local training if you don't want pushes)
        self.pushgateway_url: Optional[str] = pushgateway_url
        self.job_name: str = job_name
        self.grouping_key: Dict[str, str] = grouping_key or {}

        # runtime metadata saved after fit / drift analysis
        self.train_acc: Optional[float] = None
        self.drift_score: float = 0.0

    def fit(self):
        """
        Fit the toy pipeline on synthetic data and store train accuracy on the instance.
        Returns (accuracy, reference_dataframe).
        """
        data = pd.DataFrame(
            {
                "student": [f"S{i+1}" for i in range(10)],
                "marks": [55, 62, 71, 79, 80, 81, 85, 90, 95, 67],
            }
        )
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

        # persist training accuracy on the model instance (useful when serving)
        self.train_acc = float(acc)

        return acc, data

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        """
        Perform prediction and push metrics to Pushgateway for every invocation.

        - model_input: accepts pandas.DataFrame or convertible object (we cast to DataFrame)
        - On each call, if self.pushgateway_url is set, calls push_metrics_to_prometheus()
          with the stored train_acc and drift_score. Failures to push are caught and
          only logged so inference is not interrupted.
        """
        if self.pipeline is None:
            raise RuntimeError("Model pipeline is not fitted. Call fit() before predict().")

        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        start = time.time()
        preds = self.pipeline.predict(model_input[["marks"]].astype(float))
        elapsed = time.time() - start

        outputs = np.where(preds == 1, "Placed", "Not Placed").astype(str)

        # attempt to push metrics for every inference
        try:
            if self.pushgateway_url:
                train_acc = float(self.train_acc) if self.train_acc is not None else 0.0
                drift_score = float(getattr(self, "drift_score", 0.0))

                # Optionally you could include inference-specific metrics (latency, request_count) here:
                # For simplicity we keep the same two gauges; extend push_metrics_to_prometheus if needed.
                push_metrics_to_prometheus(
                    train_acc=train_acc,
                    drift_score=drift_score,
                    pushgateway_url=self.pushgateway_url,
                    job_name=self.job_name,
                    grouping_key=self.grouping_key,
                )
        except Exception as e:
            print(f"⚠️ push_metrics_to_prometheus in predict() raised an exception: {e}")

        return outputs


def main(args):
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    model = StudentOfferLabelModel(threshold=args.threshold, epochs=args.epochs)
    try:
        acc, reference_data = model.fit()
    except Exception as e:
        print(f"ERROR during model.fit(): {e}")
        raise

    signature = ModelSignature(
        inputs=Schema([ColSpec("double", "marks")]),
        outputs=Schema([ColSpec("string")]),
    )

    with mlflow.start_run(run_name="student_model_with_drift_check") as run:
        print(f"RUN_ID: {run.info.run_id}")

        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_metric("train_accuracy", acc)

        # Prepare current data (synthetic)
        np.random.seed(42)
        current_data = pd.DataFrame(
            {
                "student": [f"S{i+11}" for i in range(10)],
                "marks": np.random.normal(loc=82, scale=5, size=10).round().astype(int),
            }
        )
        current_data["placed"] = (current_data["marks"] > args.threshold).astype(int)

        # Build Evidently report using presets
        report = None
        try:
            report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
            report.run(reference_data=reference_data, current_data=current_data)
        except Exception as e:
            print(f"ERROR while running Evidently report: {e}")
            report = None

        # Save report if it exists and try to extract a numeric drift score
        drift_score = 0.0
        if report is not None:
            report_path = "drift_report.html"
            try:
                report.save_html(report_path)
                mlflow.log_artifact(report_path, artifact_path="reports")
                print(f"✅ Evidently report saved at {report_path} and logged to MLflow")
            except Exception as e:
                print(f"⚠️ Failed to save/log Evidently report: {e}")

            try:
                report_dict = report.as_dict()
                found = _recursive_find(report_dict, "number_of_drifted_columns")
                if found is not None:
                    drift_score = float(found)
                else:
                    alt = _recursive_find(report_dict, "drift_score") or _recursive_find(report_dict, "drift_share")
                    if alt is not None:
                        drift_score = float(alt)
                    else:
                        drift_score = 0.0
                print(f"Extracted drift_score={drift_score} from Evidently report")
            except Exception as e:
                print(f"⚠️ Could not extract drift score from Evidently report: {e}")
                drift_score = 0.0
        else:
            print("No Evidently report available; setting drift_score=0.0")

        # Save drift/train metrics on the model instance so they are serialized with the logged model
        model.train_acc = float(acc)
        model.drift_score = float(drift_score)

        # Ensure the model has pushgateway settings so served model will push on each call
        model.pushgateway_url = args.pushgateway_url
        model.job_name = "student_model_monitoring"
        model.grouping_key = {"run_id": run.info.run_id, "experiment": args.experiment_name, "model": "student_offer"}

        # Log model (note: the environment that loads the model later must have compatible deps)
        try:
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=model,
                signature=signature,
            )
            print("✅ Model logged to MLflow with embedded pushgateway config.")
        except Exception as e:
            print(f"⚠️ Failed to log model to MLflow: {e}")
            raise

        # Additionally push metrics once at training time (optional; keeps existing behavior)
        try:
            push_metrics_to_prometheus(
                train_acc=acc,
                drift_score=drift_score,
                pushgateway_url=args.pushgateway_url,
                job_name="student_model_monitoring",
                grouping_key={"run_id": run.info.run_id, "experiment": args.experiment_name, "model": "student_offer"},
            )
        except Exception:
            pass

        print(f"🔗 View in MLflow UI: {args.tracking_uri}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--tracking_uri", type=str, default="http://10.0.11.179:5000")
    parser.add_argument("--experiment_name", type=str, default="sixdee_experiments")
    parser.add_argument("--pushgateway_url", type=str, default="http://10.0.11.179:9091")
    args = parser.parse_args()

    main(args)
