import os
import pandas as pd
from pathlib import Path
from Condition2Cure.utils.helpers import *
from Condition2Cure.entities.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)

            try:
                all_schema = list(self.config.all_schema.keys())
            except AttributeError:
                all_schema = list(self.config.all_schema)

            validation_status = all(col in all_cols for col in all_schema)

            status_dict = {"Validation status": validation_status}
            
            save_json(Path(self.config.status_file), status_dict)


            return validation_status

        except Exception as e:
            raise e
