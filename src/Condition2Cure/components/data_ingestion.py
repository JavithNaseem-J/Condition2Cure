import gdown
import zipfile
from pathlib import Path
from Condition2Cure.utils.helpers import *
from Condition2Cure.entities.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            gdown.download(id=self.config.source_id, output=self.config.local_data_file, quiet=False)
            print(f"[INFO] Downloaded file: {self.config.local_data_file}")
        else:
            print(f"[INFO] File already exists: {self.config.local_data_file} ({get_size(Path(self.config.local_data_file))})")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            print(f"[INFO] Extracted files to: {unzip_path}")
