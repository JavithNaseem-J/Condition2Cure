from Condition2Cure.config.config import ConfigurationManager
from Condition2Cure.components.data_ingestion import DataIngestion

class DataIngestionPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()