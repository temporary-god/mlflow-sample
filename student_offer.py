#!/usr/bin/env python3
import argparse
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
from mlflow.data.pandas_dataset import PandasDataset


class StudentOfferLabelModel(mlflow.pyfunc.PythonModel):
    def __init__(self, threshold: float = 80.0, epochs: int = 1):
        self.pipeline = None
        self.threshold = threshold
        self.epochs = max(1, int(epochs))

    def fit(self):
        data = pd.DataFrame(
            {
                "student": [f"S{i+1}" for i in range(10)],
                "marks": [55, 62, 71, 79, 80, 81, 85, 90, 95, 67],
            }
        )
        dataset = mlflow.data.from_pandas(data, source="your_data.csv")
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
        if self.pipeline is None:
            raise RuntimeError(
                "Model pipeline is not fitted. Call fit() before predict()."
            )
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        preds = self.pipeline.predict(model_input[["marks"]].astype(float))
        return np.where(preds == 1, "Placed", "Not Placed").astype(str)


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
    pushgateway_url = _ensure_url_scheme(pushgateway_url)
    registry = CollectorRegistry()
    accuracy_metric = Gauge(
        "student_model_train_accuracy", "Training accuracy", registry=registry
    )
    drift_metric = Gauge(
        "student_model_drift_score", "Data drift score", registry=registry
    )

    accuracy_metric.set(float(train_acc))
    drift_metric.set(float(drift_score))

    try:
        # grouping_key attaches labels such as run_id/model/version
        push_to_gateway(
            pushgateway_url,
            job=job_name,
            registry=registry,
            grouping_key=grouping_key or {},
        )
        print(
            f"✅ Pushed metrics to Prometheus Pushgateway at {pushgateway_url} (job={job_name}, grouping_key={grouping_key})"
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

        # Log model (note: the environment that loads the model later must have compatible deps)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model,
            signature=signature,
        )

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
        try:
            report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
            report.run(reference_data=reference_data, current_data=current_data)
        except Exception as e:
            print(f"ERROR while running Evidently report: {e}")
            # proceed but set drift-related defaults
            report = None

        # Save report if it exists
        drift_score = 0.0
        if report is not None:
            report_path = "drift_report.html"
            try:
                report.save_html(report_path)
                mlflow.log_artifact(report_path, artifact_path="reports")
                print(f"✅ Evidently report saved at {report_path} and logged to MLflow")
            except Exception as e:
                print(f"⚠️ Failed to save/log Evidently report: {e}")

            # Extract a numeric drift score robustly (search nested dict for 'number_of_drifted_columns')
            try:
                report_dict = report.as_dict()
                found = _recursive_find(report_dict, "number_of_drifted_columns")
                if found is not None:
                    drift_score = float(found)
                else:
                    # fallback: try to find some other plausible keys, e.g., 'drift_score' or 'drift_share'
                    alt = _recursive_find(
                        report_dict, "drift_score"
                    ) or _recursive_find(report_dict, "drift_share")
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

        # Push metrics to Prometheus Pushgateway (include run id and experiment as grouping labels)
        grouping = {
            "run_id": run.info.run_id,
            "experiment": args.experiment_name,
            "model": "student_offer",
        }
        push_metrics_to_prometheus(
            train_acc=acc,
            drift_score=drift_score,
            pushgateway_url=args.pushgateway_url,
            job_name="student_model_monitoring",
            grouping_key=grouping,
        )

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
    args = parser.parse_args()

    main(args)
