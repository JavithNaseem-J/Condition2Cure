import os
import pandas as pd
from Condition2Cure import logger
from Condition2Cure.config import config
from Condition2Cure.utils.helpers import save_json


def validate_data(data_path: str = None) -> bool:

    data_path = data_path or config.raw_data_path
    
    logger.info(f"Validating data: {data_path}")
    df = pd.read_csv(data_path)
    
    # Check required columns
    missing_cols = [col for col in config.required_columns if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Save validation status
    status_dir = os.path.join(config.artifacts_root, "data_validation")
    os.makedirs(status_dir, exist_ok=True)
    
    status = {
        "valid": True,
        "columns_found": list(df.columns),
        "rows": len(df)
    }
    save_json(os.path.join(status_dir, "status.json"), status)
    
    logger.info(f"Validation passed: {len(df)} rows, {len(df.columns)} columns")
    return True



if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info(">>>>>> Stage: Data Validation started <<<<<<")
    logger.info("=" * 60)
    
    validate_data()
    
    logger.info(">>>>>> Stage: Data Validation completed <<<<<<")
