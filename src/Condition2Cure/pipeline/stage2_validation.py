from Condition2Cure.config.config import ConfigurationManager
from Condition2Cure.components.data_validation import DataValidation

class DataValidationPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()