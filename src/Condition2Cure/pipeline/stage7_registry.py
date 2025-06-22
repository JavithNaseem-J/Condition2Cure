from Condition2Cure.config.config import ConfigurationManager
from Condition2Cure.components.model_registry import ModelRegistry


class ModelRegistryPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        model_registry_config = config.get_model_registry_config()
        model_registry = ModelRegistry(config=model_registry_config)
        model_registry.registry()