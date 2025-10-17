#!/usr/bin/env python3
"""
train_and_log_model_with_path.py (metrics removed)
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
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, ColSpec
from typing import Optional
from mlflow.tracking import MlflowClient

# ---------------------
# Model class
# ---------------------
class StudentOfferLabelModel(mlflow.pyfunc.PythonModel):
    def __init__(self, threshold: float = 80.0, epochs: int = 1):
        self.pipeline = None
        self.threshold = threshold
        self.epochs = max(1, int(epochs))
        self.reference_csv_path = None

    def fit(self, reference_data: Optional[pd.DataFrame] = None):
        if reference_data is None:
            if os.path.exists("student_marks.csv"):
                reference_data = pd.read_csv("student_marks.csv")
            else:
                raise FileNotFoundError(
                    "No reference_data provided and student_marks.csv not found in cwd."
                )

        data = reference_data.copy()
        data["placed"] = (data["marks"] > self.threshold).astype(int)
        X = data[["marks"]].astype(float)
        y = data["placed"].astype(int)

        self.pipeline = Pipeline(
            [("scaler", StandardScaler()), ("lr", LogisticRegression())]
        )

        for _ in range(self.epochs):
            self.pipeline.fit(X, y)

        # Return trained data for artifact logging
        return data

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model pipeline is not fitted. Call fit() before predict().")
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        preds = self.pipeline.predict(model_input[["marks"]].astype(float))
        return np.where(preds == 1, "Placed", "Not Placed").astype(str)

    def load_context(self, context):
        try:
            self.reference_csv_path = context.artifacts.get("data/student_marks.csv")
        except Exception:
            self.reference_csv_path = None

        if self.reference_csv_path and os.path.exists(self.reference_csv_path):
            try:
                _ = pd.read_csv(self.reference_csv_path)
            except Exception:
                pass


# ---------------------
# Helpers
# ---------------------
def _ensure_url_scheme(url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return "http://" + url


# ---------------------
# Main training flow
# ---------------------
def main(args):
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    if not os.path.isfile(args.reference_path):
        raise FileNotFoundError(f"Reference CSV not found at: {args.reference_path}")

    reference_data = pd.read_csv(args.reference_path)
    model = StudentOfferLabelModel(threshold=args.threshold, epochs=args.epochs)

    # Start MLflow run
    if mlflow.active_run() is None:
        run_cm = mlflow.start_run(run_name="student_model_no_metrics")
    else:
        run_cm = mlflow.start_run(run_name="student_model_no_metrics", nested=True)

    with run_cm as run:
        print(f"RUN_ID: {run.info.run_id}")

        # Fit model
        ref_df = model.fit(reference_data=reference_data)

        # Signature
        signature = ModelSignature(
            inputs=Schema([ColSpec("double", "marks")]),
            outputs=Schema([ColSpec("string", "prediction")]),
        )

        # Log params (no metrics)
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("epochs", args.epochs)

        # Log reference CSV as artifact
        try:
            #mlflow.log_artifact(args.reference_path, artifact_path="data")
            print(f"✅ Logged reference CSV as run artifact: {args.reference_path}")
        except Exception as e:
            print(f"[WARN] Failed to log reference CSV as run artifact: {e}")

        artifacts = {"data/student_marks.csv": args.reference_path}

        # Log model
        try:
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=model,
                signature=signature,
                artifacts=artifacts,
            )
            print("✅ Logged pyfunc model and included reference CSV in artifacts")
        except Exception as e:
            print(f"[ERROR] Failed to log model: {e}")
            raise

        # Optionally register model in MLflow Model Registry
        if args.register_model:
            model_uri = f"runs:/{run.info.run_id}/model"
            model_name = args.model_name or "student_offer"
            try:
                print(f"[INFO] Registering model {model_name} from {model_uri}")
                mv = mlflow.register_model(model_uri=model_uri, name=model_name)
                client = MlflowClient(tracking_uri=args.tracking_uri)
                client.set_model_version_tag(
                    name=model_name, version=mv.version, key="run_id", value=run.info.run_id
                )
                client.set_model_version_tag(
                    name=model_name,
                    version=mv.version,
                    key="mlflow_registered_by",
                    value=os.getenv("USER", "unknown"),
                )
                print(f"✅ Registered model {model_name}, version={mv.version} and tagged with run_id")
            except Exception as e:
                print(f"[WARN] Model registration failed: {e}")

        # MLflow UI link
        try:
            print(f"🔗 View in MLflow UI: {args.tracking_uri}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--tracking_uri", type=str, default="http://10.0.11.179:5001")
    parser.add_argument("--experiment_name", type=str, default="sixdee_experiments")
    #parser.add_argument("--reference_path", type=str, required=True)
    parser.add_argument("--register_model", type=lambda s: s.lower() in ["1", "true", "yes"], default=False)
    parser.add_argument("--model_name", type=str, default="student_offer")
    parser.add_argument("--reference_path", type=str, default="/home/coder/project/sample/student_marks.csv")
    args = parser.parse_args()

    main(args)
