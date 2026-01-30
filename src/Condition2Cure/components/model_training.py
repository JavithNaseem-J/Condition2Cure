import os
import numpy as np
import mlflow
import optuna
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from Condition2Cure import logger
from Condition2Cure.config import config
from Condition2Cure.utils.helpers import save_json, save_model


def train_model(n_trials: int = None) -> dict:
    """
    Train XGBoost model with Optuna hyperparameter tuning.
    
    Uses ONLY training data for cross-validation.
    Test data is never touched here.
    
    Args:
        n_trials: Number of Optuna trials (uses config default if None)
    
    Returns:
        Dictionary with training info
    """
    n_trials = n_trials or config.optuna_trials
    
    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Condition2Cure")
    
    # Load TRAINING data only
    logger.info("Loading training data...")
    X_train = np.load(config.features_path)
    y_train = np.load(config.labels_path)
    logger.info(f"Training data: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Number of classes: {len(np.unique(y_train))}")
    
    # Define Optuna objective
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "verbosity": 0,
            "random_state": config.random_state,
            "tree_method": "hist",
            "n_jobs": -1
        }
        
        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
        scores = cross_val_score(model, X_train, y_train, scoring="f1_weighted", cv=cv, n_jobs=-1)
        
        return scores.mean()
    
    # Run Optuna optimization
    logger.info(f"Starting hyperparameter tuning ({n_trials} trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_score = study.best_value
    
    logger.info(f"Best CV F1 Score: {best_score:.4f}")
    logger.info(f"Best Parameters: {best_params}")
    
    # Train final model with best parameters
    logger.info("Training final model on full training set...")
    best_params["verbosity"] = 0
    best_params["random_state"] = config.random_state
    best_params["n_jobs"] = -1
    
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)
    
    # Save model
    save_model(final_model, config.model_path)
    
    # Save training info
    model_info = {
        "best_cv_f1_score": float(best_score),
        "best_params": best_params,
        "n_train_samples": int(X_train.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_classes": int(len(np.unique(y_train)))
    }
    
    info_path = os.path.join(os.path.dirname(config.model_path), "training_info.json")
    save_json(info_path, model_info)
    
    # Log to MLflow
    with mlflow.start_run(run_name="XGBoost_Training"):
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_f1_score", best_score)
        mlflow.log_metric("n_train_samples", X_train.shape[0])
        mlflow.sklearn.log_model(final_model, "model")
    
    logger.info("Training complete!")
    return model_info


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info(">>>>>> Stage: Model Training started <<<<<<")
    logger.info("=" * 60)
    
    info = train_model()
    
    logger.info(f"CV F1 Score: {info['best_cv_f1_score']:.4f}")
    logger.info(">>>>>> Stage: Model Training completed <<<<<<")
