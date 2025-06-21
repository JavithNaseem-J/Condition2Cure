import os
import joblib
import mlflow
import optuna
import numpy as np
import dagshub
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from Condition2Cure.utils.helpers import save_json
from Condition2Cure import logger
from Condition2Cure.entities.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Condition2Cure")
        logger.info("MLflow tracking setup complete.")

    def train(self):
        logger.info("Loading training data...")
        
        X = np.load(self.config.features_path, allow_pickle=True)
        y = np.load(self.config.labels_path, allow_pickle=True)
        
        logger.info(f"Data loaded. Features shape: {X.shape}, Labels shape: {y.shape}")

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "use_label_encoder": False,
                "verbosity": 0,
                "eval_metric": "mlogloss",
                "random_state": self.config.random_state
            }

            model = XGBClassifier(**params)
            scores = cross_val_score(
                model, X, y,
                scoring="accuracy",
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.random_state)            )
            return scores.mean()

        logger.info("Running Optuna hyperparameter tuning for XGBoost...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=2, n_jobs=-1)

        best_params = study.best_params
        best_score = study.best_value

        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best CV Score: {best_score}")

        # Train final model with best parameters
        best_model = XGBClassifier(**best_params)
        best_model.fit(X, y)

        # Save model - Fixed: Use the correct path structure
        os.makedirs(self.config.root_dir, exist_ok=True)
        joblib.dump(best_model, self.config.model_path)
        logger.info(f"Model saved to: {self.config.model_path}")

        best_model_info = {
            "best_score": float(best_score), 
            "best_params": best_params
        }
        best_info_path = os.path.join(self.config.root_dir, "best_model_info.json")
        save_json(Path(best_info_path), best_model_info)
        logger.info(f"Best model info saved to: {best_info_path}")

        # Log to MLflow
        with mlflow.start_run(run_name="XGBoost_Hyperparameter_Tuning"):
            # Log best parameters
            mlflow.log_params(best_params)
            mlflow.log_metric("best_cv_score", best_score)
            mlflow.log_metric("n_features", X.shape[1])
            mlflow.log_metric("n_samples", X.shape[0])
            mlflow.log_metric("n_classes", len(np.unique(y)))
            
            # Log model
            mlflow.sklearn.log_model(
                best_model, 
                artifact_path="model", 
                registered_model_name="Condition2CureModel"
            )
            
            # Log artifacts
            mlflow.log_artifact(best_info_path)
            mlflow.log_artifact(self.config.model_path)
            
            # Add tags
            mlflow.set_tag("model_type", "XGBoost")
            mlflow.set_tag("optimization", "Optuna")
            mlflow.set_tag("stage", "training")
            
            logger.info("Model and metrics logged to MLflow.")

        logger.info(f"Model training complete. Best CV score: {best_score:.4f}")
        return best_model_info