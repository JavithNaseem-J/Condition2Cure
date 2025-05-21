from Condition2Cure.config.config import ConfigurationManager
from Condition2Cure.components.feature_engineering import FeatureEngineering

class FeatureEngineeringPipeline:
    def __init__(aelf):
        pass

    def run(self):
        config = ConfigurationManager()
        feature_engineering_config = config.get_feature_engineering_config()
        feature_engineering = FeatureEngineering(config=feature_engineering_config)
        feature_engineering.transfom()