import json
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from Condition2Cure.entities.config_entity import ModelRegistryConfig
from Condition2Cure import logger

class ModelRegistry:
    def __init__(self, config: ModelRegistryConfig):
        self.config = config
        self.client = MlflowClient()

    def load_metric(self) -> float:
        with open(self.config.metric_path, "r") as f:
            metrics = json.load(f)
        return float(metrics.get(self.config.metric_key))

    def get_latest_model_by_stage(self, stage: str):
        try:
            versions = self.client.get_latest_versions(name=self.config.model_name, stages=[stage])
            return versions[0] if versions else None
        except MlflowException:
            # Model not registered yet
            return None

    def promote_model(self, version):
        self.client.transition_model_version_stage(
            name=self.config.model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        logger.info(f"Promoted version {version} to Production.")

    def registry(self):
        logger.info("Running model registry promotion check...")
        new_score = self.load_metric()
        staging_model = self.get_latest_model_by_stage("Staging")
        if not staging_model:
            logger.warning("No staging model found. Skipping promotion check.")
            logger.info("Register a model to 'Staging' stage first to enable promotion workflow.")
            return

        prod_model = self.get_latest_model_by_stage("Production")
        prod_score = None
        if prod_model:
            run_id = prod_model.run_id
            prod_metrics = self.client.get_run(run_id).data.metrics
            prod_score = float(prod_metrics.get(self.config.metric_key, 0))

        logger.info(f"Staging {self.config.metric_key}: {new_score}")
        logger.info(f"Production {self.config.metric_key}: {prod_score}")

        if prod_score is None or new_score > prod_score:
            self.promote_model(staging_model.version)
        else:
            logger.info("No promotion. Staging model is not better than Production.")