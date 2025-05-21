import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from Condition2Cure.entities.config_entity import ModelTrainerConfig
from Condition2Cure import logger

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        logger.info("Loading features...")
        data = np.load(self.config.features_path, allow_pickle=True).item()
        X, y = data["X"], data["y"]

        logger.info("Splitting train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )

        logger.info("Training PassiveAggressiveClassifier ...")
        model = PassiveAggressiveClassifier(max_iter=self.config.max_iter)
        model.fit(X_train, y_train)

        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
        joblib.dump(model, self.config.model_path)
        logger.info(f"Model saved to {self.config.model_path}")
