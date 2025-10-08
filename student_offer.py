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
  --register_model True \
  --model_name evidently-metrics-student-offer
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
from typing import Any, Dict, Optional
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

        preds = self.pipeline.predict(X)
        acc = accuracy_score(y, preds)
        return acc, data

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


def _recursive_find(obj: Any, target_key: str) -> Optional[Any]:
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
        run_cm = mlflow.start_run(run_name="student_model_with_drift_check")
    else:
        run_cm = mlflow.start_run(run_name="student_model_with_drift_check", nested=True)

    with run_cm as run:
        print(f"RUN_ID: {run.info.run_id}")

        # Fit model
        acc, ref_df = model.fit(reference_data=reference_data)

        # Signature
        signature = ModelSignature(
            inputs=Schema([ColSpec("double", "marks")]),
            outputs=Schema([ColSpec("string", "prediction")]),
        )

        # Log params & metrics
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_metric("train_accuracy", acc)

        # Evidently drift evaluation (best-effort)
        drift_score_percent = 0.0
        try:
            report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
            # Use ref_df (trained-on) and a small synthetic current_data for quick drift check
            np.random.seed(42)
            current_data = pd.DataFrame(
                {
                    "student": [f"S{i+11}" for i in range(10)],
                    "marks": np.random.normal(loc=82, scale=5, size=10).round().astype(int),
                }
            )
            current_data["placed"] = (current_data["marks"] > args.threshold).astype(int)

            report.run(reference_data=ref_df, current_data=current_data)

            report.save_html("drift_report.html")
            mlflow.log_artifact("drift_report.html", artifact_path="reports")
            print("✅ Evidently report saved and logged")

            report_dict = report.as_dict()
            num_drifted = _recursive_find(report_dict, "number_of_drifted_columns")
            total_cols = _recursive_find(report_dict, "number_of_columns") or len(ref_df.columns) or None
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

            drift_score_percent = float(drift_fraction) * 100.0
            print(f"Extracted drift fraction={drift_fraction}, percent={drift_score_percent}")
        except Exception as e:
            print(f"[WARN] Evidently drift check failed: {e}")
            drift_score_percent = 0.0

        # Log drift metric to MLflow as well
        mlflow.log_metric("drift_score_percent", drift_score_percent)

        # Log reference CSV as artifact and bundle into model artifacts
        try:
            mlflow.log_artifact(args.reference_path, artifact_path="data")
            print(f"✅ Logged reference CSV as run artifact: {args.reference_path}")
        except Exception as e:
            print(f"[WARN] Failed to log reference CSV as run artifact: {e}")

        artifacts = {"data/student_marks.csv": args.reference_path}

        # Log pyfunc model
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

        # Optionally register model into Model Registry and tag version with run_id
        if args.register_model:
            model_uri = f"runs:/{run.info.run_id}/model"
            model_name = args.model_name or "student_offer"
            try:
                print(f"[INFO] Registering model {model_name} from {model_uri}")
                # mlflow.register_model returns a ModelVersion object in newer mlflow clients
                mv = mlflow.register_model(model_uri=model_uri, name=model_name)
                # Wait/ensure registration metadata may take some time depending on MLflow setup.
                client = MlflowClient(tracking_uri=args.tracking_uri)
                mv_number = mv.version if hasattr(mv, "version") else str(mv)  # defensive
                # Add run_id tag on model version so deploy DAG can find mapping
                try:
                    client.set_model_version_tag(name=model_name, version=mv.version, key="run_id", value=run.info.run_id)
                    client.set_model_version_tag(name=model_name, version=mv.version, key="mlflow_registered_by", value=os.getenv("USER", "unknown"))
                    print(f"✅ Registered model {model_name}, version={mv.version} and tagged with run_id")
                except Exception as e:
                    print(f"[WARN] Model registered but failed to tag version: {e}")
            except Exception as e:
                print(f"[WARN] Model registration failed: {e}")

        # print MLflow UI link
        try:
            print(f"🔗 View in MLflow UI: {args.tracking_uri}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
        except Exception:
            pass

        # End run context
    # end with run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--tracking_uri", type=str, default="http://10.0.11.179:5001")
    parser.add_argument("--experiment_name", type=str, default="sixdee_experiments")
    parser.add_argument("--pushgateway_url", type=str, default=None)  # kept for compatibility, unused
    parser.add_argument("--image_name", type=str, default=os.environ.get("IMAGE_NAME", "student-offer"))
    parser.add_argument("--image_version", type=str, default=os.environ.get("IMAGE_VERSION", "dev"))
    parser.add_argument("--reference_path", type=str, required=True)
    parser.add_argument("--register_model", type=lambda s: s.lower() in ["1", "true", "yes"], default=False,
                        help="If true, register the model in MLflow Model Registry")
    parser.add_argument("--model_name", type=str, default="student_offer", help="Registered model name (if --register_model True)")
    args = parser.parse_args()

    main(args)

