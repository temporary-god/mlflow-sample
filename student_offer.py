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
from evidently.metrics import DataDriftTable, ClassificationPerformanceMetric
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


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
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        preds = self.pipeline.predict(model_input[["marks"]].astype(float))
        return np.where(preds == 1, "Placed", "Not Placed").astype(str)


def push_metrics_to_prometheus(train_acc, drift_score, pushgateway_url, job_name):
    registry = CollectorRegistry()
    accuracy_metric = Gauge(
        "student_model_train_accuracy", "Training accuracy", registry=registry
    )
    drift_metric = Gauge(
        "student_model_drift_score", "Data drift score", registry=registry
    )

    accuracy_metric.set(train_acc)
    drift_metric.set(drift_score)

    push_to_gateway(pushgateway_url, job=job_name, registry=registry)
    print(f"✅ Pushed metrics to Prometheus at {pushgateway_url}")


def main(args):
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    model = StudentOfferLabelModel(threshold=args.threshold, epochs=args.epochs)
    acc, reference_data = model.fit()

    run_name = "student_model_with_drift_check"
    signature = ModelSignature(
        inputs=Schema([ColSpec("double", "marks")]),
        outputs=Schema([ColSpec("string")]),
    )

    with mlflow.start_run(run_name=run_name) as run:
        print(f"RUN_ID: {run.info.run_id}")

        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_metric("train_accuracy", acc)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model,
            signature=signature,
        )

        np.random.seed(42)
        current_data = pd.DataFrame(
            {
                "student": [f"S{i+11}" for i in range(10)],
                "marks": np.random.normal(loc=82, scale=5, size=10).round().astype(int),
            }
        )
        current_data["placed"] = (current_data["marks"] > args.threshold).astype(int)

        report = Report(metrics=[DataDriftTable(), ClassificationPerformanceMetric()])
        report.run(reference_data=reference_data, current_data=current_data)

        report_path = "drift_report.html"
        report.save_html(report_path)
        mlflow.log_artifact(report_path, artifact_path="reports")

        print("✅ Model trained and drift report logged")

        # Extract drift score from report (example using DataDriftTable)
        drift_score = report.as_dict()["metrics"][0]["result"][
            "number_of_drifted_columns"
        ]

        # Push metrics to Prometheus
        push_metrics_to_prometheus(
            train_acc=acc,
            drift_score=float(drift_score),
            pushgateway_url=args.pushgateway_url,
            job_name="student_model_monitoring",
        )

        print(
            f"🔗 View in MLflow UI: {args.tracking_uri}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--tracking_uri", type=str, default="http://localhost:5000")
    parser.add_argument(
        "--experiment_name", type=str, default="student_drift_monitoring"
    )
    parser.add_argument("--pushgateway_url", type=str, default="localhost:9091")
    args = parser.parse_args()

    main(args)
