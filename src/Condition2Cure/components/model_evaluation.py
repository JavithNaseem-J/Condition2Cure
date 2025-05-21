import os
import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)
from sklearn.model_selection import train_test_split
from Condition2Cure.entities.config_entity import ModelEvaluationConfig
from Condition2Cure import logger


class ModelEvaluator:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def _load(self):
        data = np.load(self.config.features_path, allow_pickle=True).item()
        X, y = data["X"], data["y"]
        model = joblib.load(self.config.model_path)
        label_encoder = joblib.load(self.config.label_encoder_path)
        return X, y, model, label_encoder

    def _evaluate(self, model, X, y, label: str, plot_confusion=False, plot_roc=False):
        logger.info(f"Evaluating on {label} set...")

        y_pred = model.predict(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y, y_pred, average="weighted", zero_division=0)
        }

        if plot_confusion:
            self._plot_confusion_matrix(y, y_pred)

        
        return metrics

    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(self.config.cm_path)
        plt.close()


    def _save_metrics(self, metrics):
        os.makedirs(self.config.root_dir, exist_ok=True)
        with open(self.config.metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved metrics to: {self.config.metrics_path}")

    def evaluator(self):
        X, y, model, label_encoder = self._load()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )

        metrics = {
            "train": self._evaluate(model, X_train, y_train, label="train"),
            "test": self._evaluate(model, X_test, y_test, label="test", plot_confusion=True, plot_roc=True)
        }

        self._save_metrics(metrics)
        logger.info("✅ Evaluation completed test sets.")
