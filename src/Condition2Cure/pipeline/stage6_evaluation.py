from Condition2Cure.config.config import ConfigurationManager
from Condition2Cure.components.model_evaluation import ModelEvaluator


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        model_evaluator_config = config.get_model_evaluation_config()
        model_evaluator = ModelEvaluator(config=model_evaluator_config)
        model_evaluator.evaluation()