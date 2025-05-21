import os
import argparse
from Condition2Cure import logger
from Condition2Cure.pipeline.stage1_ingestion import DataIngestionPipeline
from Condition2Cure.pipeline.stage2_validation import DataValidationPipeline
from Condition2Cure.pipeline.stage3_cleaning import DataCleaningPipeline
from Condition2Cure.pipeline.stage4_feature import FeatureEngineeringPipeline
from Condition2Cure.pipeline.stage5_training import ModelTrainingPipeline
from Condition2Cure.pipeline.stage6_evaluation import ModelEvaluationPipeline




def run_stage(stage_name):
    logger.info(f">>>>>> Stage {stage_name} started <<<<<<")

    try:
        if stage_name == "data_ingestion":
            stage = DataIngestionPipeline()
            stage.run()

        elif stage_name == "data_validation":
            stage = DataValidationPipeline()
            stage.run()


        elif stage_name == "data_cleaning":
            stage = DataCleaningPipeline()
            stage.run()

        elif stage_name == "feature_engineering":
            stage = FeatureEngineeringPipeline()
            stage.run()

        elif stage_name == "model_training":
            stage = ModelTrainingPipeline()
            stage.run()

        elif stage_name == "model_evaluation":
            stage = ModelEvaluationPipeline()
            stage.run()

        else:
            raise ValueError(f"Unknown stage: {stage_name}")

        logger.info(f">>>>>> Stage {stage_name} completed <<<<<<\n\nx==========x")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except KeyError as e:
        logger.error(f"Missing key in configuration or data: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run specific pipeline stage.")
    parser.add_argument("--stage", help="Name of the stage to run")
    args = parser.parse_args()

    if args.stage:
        run_stage(args.stage)
    else:
        stages = [
            "data_ingestion",
            "data_validation",
            "data_cleaning",
            "feature_engineering",
            "model_training",
            "model_evaluation",
        ]
        for stage in stages:
            run_stage(stage)


