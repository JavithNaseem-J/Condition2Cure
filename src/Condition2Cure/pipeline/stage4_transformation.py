from Condition2Cure.config.config import ConfigurationManager
from Condition2Cure.components.data_transformation import DataTransformation


class DataTransformationPipeline:
    def __init__(aelf):
        pass

    def run(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        feature_engineering = DataTransformation(config=data_transformation_config)
        feature_engineering.transform()