from Condition2Cure.config.data_config import DataConfigurationManager
from Condition2Cure.components.data_ingestion import DataIngestion
from Condition2Cure.components.data_validation import DataValidation
from Condition2Cure.components.data_cleaning import DataCleaning
from Condition2Cure.components.data_transformation import DataTransformation


class FeaturePipeline:
    def __init__(self):
        pass

    def run(self):
        config = DataConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()

        data_cleaning_config = config.get_data_cleaning_config()
        data_cleaning = DataCleaning(config=data_cleaning_config)
        data_cleaning.clean()

        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.transform()
