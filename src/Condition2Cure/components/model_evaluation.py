"""
Model Evaluation
================
Evaluate model on held-out TEST set only.
No data leakage - test set was split during transformation.

Run: python -m Condition2Cure.components.model_evaluation
"""
import os
import numpy as np
import joblib
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from Condition2Cure import logger
from Condition2Cure.config import config
from Condition2Cure.utils.helpers import save_json, load_model


def evaluate_model() -> dict:
    """
    Evaluate model on held-out test set.
    
    IMPORTANT: Uses test set that was split during transformation.
    This is the ONLY place the test set is used.
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Load test data (held out during transformation)
    logger.info("Loading held-out test data...")
    X_test = np.load(config.test_features_path)
    y_test = np.load(config.test_labels_path)
    logger.info(f"Test data: X={X_test.shape}, y={y_test.shape}")
    
    # Load model and label encoder
    logger.info("Loading model and label encoder...")
    model = load_model(config.model_path)
    label_encoder = joblib.load(config.label_encoder_path)
    
    # Predictions
    logger.info("Generating predictions on test set...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted")),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "n_test_samples": int(len(y_test))
    }
    
    logger.info("=" * 50)
    logger.info("TEST SET METRICS (Held-out data)")
    logger.info("=" * 50)
    for metric, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")
    logger.info("=" * 50)
    
    # Save metrics
    os.makedirs(os.path.dirname(config.metrics_path), exist_ok=True)
    save_json(config.metrics_path, metrics)
    
    # Create and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(config.confusion_matrix_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved: {config.confusion_matrix_path}")
    
    # Log to MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Condition2Cure")
    
    with mlflow.start_run(run_name="Model_Evaluation"):
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.set_tag("stage", "evaluation")
        mlflow.set_tag("evaluation_date", str(pd.Timestamp.now()))
        mlflow.log_artifact(config.metrics_path)
        mlflow.log_artifact(config.confusion_matrix_path)
    
    logger.info("Model evaluation complete!")
    return metrics


# ============================================
# DVC Entry Point
# ============================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info(">>>>>> Stage: Model Evaluation started <<<<<<")
    logger.info("=" * 60)
    
    metrics = evaluate_model()
    
    logger.info(f"Test F1 Score: {metrics['f1_weighted']:.4f}")
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(">>>>>> Stage: Model Evaluation completed <<<<<<")