import os
import pandas as pd
from Condition2Cure import logger
from Condition2Cure.config import config
from Condition2Cure.utils.nlp_utils import clean_text


def clean_data(input_path: str = None, output_path: str = None) -> pd.DataFrame:

    input_path = input_path or config.raw_data_path
    output_path = output_path or config.cleaned_data_path
    
    logger.info(f"Loading raw data: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Raw data: {len(df)} rows")
    
    # Filter to selected conditions (from config, not hardcoded!)
    df = df[df[config.target_column].isin(config.conditions)]
    logger.info(f"After filtering conditions: {len(df)} rows")
    
    # Validate required columns exist
    if config.text_column not in df.columns:
        raise ValueError(f"Text column '{config.text_column}' not found")
    if config.target_column not in df.columns:
        raise ValueError(f"Target column '{config.target_column}' not found")
    
    # Clean text
    logger.info("Cleaning review texts...")
    df['clean_review'] = df[config.text_column].astype(str).apply(clean_text)
    
    # Remove empty reviews
    df = df[df['clean_review'].str.len() > 0]
    df = df.dropna(subset=['clean_review', config.target_column])
    
    logger.info(f"After cleaning: {len(df)} rows")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data: {output_path}")
    
    return df


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info(">>>>>> Stage: Data Cleaning started <<<<<<")
    logger.info("=" * 60)
    
    clean_data()
    
    logger.info(">>>>>> Stage: Data Cleaning completed <<<<<<")