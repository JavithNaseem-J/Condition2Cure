import os
import gdown
import zipfile
from Condition2Cure import logger
from Condition2Cure.config import config


def download_data() -> str:
    """Download data from Google Drive if not exists."""
    local_file = os.path.join(config.artifacts_root, "data_ingestion", "data.zip")
    unzip_dir = os.path.join(config.artifacts_root, "data_ingestion")
    
    os.makedirs(unzip_dir, exist_ok=True)
    
    # Download if not exists
    if not os.path.exists(local_file):
        logger.info("Downloading data from Google Drive...")
        gdown.download(id=config.data_source_id, output=local_file, quiet=False)
        logger.info(f"Downloaded: {local_file}")
    else:
        logger.info(f"Data already exists: {local_file}")
    
    # Extract
    if not os.path.exists(config.raw_data_path):
        logger.info("Extracting zip file...")
        with zipfile.ZipFile(local_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
        logger.info(f"Extracted to: {unzip_dir}")
    
    return config.raw_data_path



if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info(">>>>>> Stage: Data Ingestion started <<<<<<")
    logger.info("=" * 60)
    
    download_data()
    
    logger.info(">>>>>> Stage: Data Ingestion completed <<<<<<")
