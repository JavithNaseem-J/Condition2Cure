from Condition2Cure.config.config import ConfigurationManager
from Condition2Cure.components.data_cleaning import DataCleaning


class DataCleaningPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        data_cleaning_config = config.get_data_cleaning_config()
        data_cleaning = DataCleaning(config=data_cleaning_config)
        data_cleaning.clean()