from pathlib import Path
from src.Condition2Cure.constants import *
from Condition2Cure.utils.helpers import *
from Condition2Cure.entities.config_entity import DataIngestionConfig, DataValidationConfig, DataCleaningConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig



class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_id=config.source_id,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.columns
        
        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            status_file=config.status_file,
            unzip_data_dir=config.unzip_data_dir,
            all_schema=schema,
        )
        return data_validation_config
    

    def get_data_cleaning_config(self) -> DataCleaningConfig:
        config = self.config.data_cleaning
        create_directories([config.root_dir])
        
        data_cleaning_config =  DataCleaningConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            cleaned_data_path=config.cleaned_data_path
        )

        return data_cleaning_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.feature_engineering
        params = self.params.vectorizer
        schema = self.schema.target_column

        create_directories([config.root_dir])

        data_transformation_config =  DataTransformationConfig(
            root_dir=config.root_dir,
            cleaned_data_path=config.cleaned_data_path,
            vectorizer_path=config.vectorizer_path,
            features_path=config.features_path,
            labels_path=config.labels_path,
            label_encoder_path=config.label_encoder_path,
            target_column=schema.name,
            max_features=params.max_features,
            ngram_range=params.ngram_range,
            svd_path=config.svd_path,
            svd_components=params.svd_components
        )

        return data_transformation_config

    
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