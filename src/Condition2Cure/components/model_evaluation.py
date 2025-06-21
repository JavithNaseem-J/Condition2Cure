import os
import json
import pandas as pd
import numpy as np
import joblib
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from Condition2Cure.utils.helpers import *
from Condition2Cure import logger
from Condition2Cure.entities.config_entity import ModelEvaluationConfig


class ModelEvaluator:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

        mlflow.set_tracking_uri("./mlruns")
        mlflow.set_experiment("Condition2Cure")
        logger.info("MLflow tracking setup complete.")

    def evaluation(self):
        # Load data and artifacts
        if not os.path.exists(self.config.features_path):
            raise FileNotFoundError(f"Features file not found: {self.config.features_path}")
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model file not found: {self.config.model_path}")
        if not os.path.exists(self.config.label_encoder_path):
            raise FileNotFoundError(f"Label encoder file not found: {self.config.label_encoder_path}")

        X = np.load(self.config.features_path, allow_pickle=True)
        y = np.load(self.config.labels_path, allow_pickle=True)

        model = joblib.load(self.config.model_path)
        label_encoder = joblib.load(self.config.label_encoder_path)

        logger.info("Loaded features, model, and label encoder.")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )

        logger.info("Split data into train and test sets.")

        # Predictions and metrics
        preds = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision_weighted": precision_score(y_test, preds, average="weighted"),
            "recall_weighted": recall_score(y_test, preds, average="weighted"),
            "f1_weighted": f1_score(y_test, preds, average="weighted"),
            "precision_macro": precision_score(y_test, preds, average="macro"),
            "recall_macro": recall_score(y_test, preds, average="macro"),
            "f1_macro": f1_score(y_test, preds, average="macro"),
        }

        create_directories([os.path.dirname(self.config.metrics_path)])
        save_json(path=Path(self.config.metrics_path), data=metrics)
        logger.info(f"Metrics saved to {self.config.metrics_path}")

        # MLflow logging
        with mlflow.start_run(run_name="Model Evaluation"):
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            mlflow.set_tag("stage", "evaluation")
            mlflow.set_tag("evaluation_date", str(pd.Timestamp.now()))
            mlflow.log_artifact(self.config.metrics_path)
            mlflow.log_artifact(self.config.model_path)
            mlflow.log_artifact(self.config.label_encoder_path)
            logger.info("Metrics and artifacts logged to MLflow.")

            cm = confusion_matrix(y_test, preds)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            
            cm_save_path = os.path.join(os.path.dirname(self.config.cm_path), "cm.png")
            create_directories([os.path.dirname(cm_save_path)])
            plt.savefig(cm_save_path)
            plt.close()
            mlflow.log_artifact(cm_save_path)
            logger.info(f"Confusion matrix saved to {cm_save_path}")

        logger.info("Model evaluation complete. Metrics and plots logged.")
        return metrics