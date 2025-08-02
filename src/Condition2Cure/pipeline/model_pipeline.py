from Condition2Cure.config.model_config import ModelConfigurationManager
from Condition2Cure.components.model_training import ModelTrainer
from Condition2Cure.components.model_evaluation import ModelEvaluation
from Condition2Cure.components.model_registry import ModelRegistry

class ModelPipeline:
    def __init__(self):
        pass

    
    def run(self):
        config = ModelConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()

        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.evaluation()

        model_registry_config = config.get_model_registry_config()
        model_registry = ModelRegistry(config=model_registry_config)
        model_registry.registry()

