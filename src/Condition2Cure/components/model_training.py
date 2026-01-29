import os
import joblib
import mlflow
import optuna
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from Condition2Cure.utils.helpers import save_json
from Condition2Cure import logger


class ModelTrainer:

    
    def __init__(self, config):

        self.config = config
        
        # Setup MLflow for experiment tracking
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Condition2Cure")

    def train(self, n_trials: int = 2):

        # Step 1: Load the data
        logger.info("Loading training data...")
        X = np.load(self.config.features_path)
        y = np.load(self.config.labels_path)
        logger.info(f"Data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Number of classes: {len(np.unique(y))}")

        # Step 2: Define the objective function for Optuna
        def objective(trial):

            # Suggest hyperparameters to try
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 150),  # Reduced range
                "max_depth": trial.suggest_int("max_depth", 3, 7),  # Reduced max depth
                "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "verbosity": 0,
                "random_state": 42,
                "tree_method": "hist"  # Faster histogram-based algorithm
            }

            # Create model and evaluate with cross-validation
            model = XGBClassifier(**params, n_jobs=-1)  # Use all CPU cores for XGBoost
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced from 5 to 3 folds
            scores = cross_val_score(model, X, y, scoring="f1_weighted", cv=cv, n_jobs=-1)  # Parallelize CV
            
            return scores.mean()

        # Step 3: Run Optuna optimization
        logger.info(f"Starting hyperparameter tuning ({n_trials} trials)...")
        
        # Suppress Optuna logs for cleaner output
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"Best F1 Score: {best_score:.4f}")
        logger.info(f"Best Parameters: {best_params}")

        # Step 4: Train final model with best parameters
        logger.info("Training final model...")
        best_params["verbosity"] = 0
        best_params["random_state"] = 42
        
        final_model = XGBClassifier(**best_params)
        final_model.fit(X, y)

        # Step 5: Save the model
        os.makedirs(self.config.root_dir, exist_ok=True)
        joblib.dump(final_model, self.config.model_path)
        logger.info(f"Model saved to: {self.config.model_path}")

        # Save model info
        model_info = {
            "best_f1_score": float(best_score),
            "best_params": best_params,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(len(np.unique(y)))
        }
        
        info_path = os.path.join(self.config.root_dir, "best_model_info.json")
        save_json(Path(info_path), model_info)

        # Step 6: Log to MLflow
        with mlflow.start_run(run_name="XGBoost_Training"):
            mlflow.log_params(best_params)
            mlflow.log_metric("f1_score", best_score)
            mlflow.log_metric("n_samples", X.shape[0])
            mlflow.sklearn.log_model(final_model, "model")
            
        logger.info("Training complete!")
        return model_info
