from pathlib import Path
from src.Condition2Cure.constants import *
from Condition2Cure.utils.helpers import *
from Condition2Cure.entities.config_entity import ModelTrainerConfig, ModelEvaluationConfig, ModelRegistryConfig


class ModelConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_training
        params = self.params.vectorizer
        split = self.params.train_test_split

        create_directories([config.root_dir])

        model_training_config =  ModelTrainerConfig(
            root_dir=config.root_dir,
            features_path=config.features_path,
            labels_path=config.labels_path,
            model_path=config.model_path,
            max_iter=params.max_iter,
            test_size=split.test_size,
            random_state=split.random_state
        )

        return model_training_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        split = self.params.train_test_split

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            features_path=config.features_path,
            labels_path=config.labels_path,
            model_path=config.model_path,
            label_encoder_path=config.label_encoder_path,
            test_size=split.test_size,
            random_state=split.random_state,
            metrics_path=config.metrics_path,
            cm_path=config.cm_path
        )

        return model_evaluation_config
    


    def get_model_registry_config(self) -> ModelRegistryConfig:
        config = self.config.model_registry


        model_registry_config =  ModelRegistryConfig(
            model_name=config.model_name,
            metric_path=config.metric_path,
            metric_key=config.metric_key
        )

        return model_registry_config