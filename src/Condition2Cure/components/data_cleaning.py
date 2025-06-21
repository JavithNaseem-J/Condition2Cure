import os
import pandas as pd
from Condition2Cure.utils.helpers import create_directories
from Condition2Cure.entities.config_entity import DataCleaningConfig
from Condition2Cure.utils.nlp_utils import clean_text
from Condition2Cure import logger


class DataCleaning:
    def __init__(self, config: DataCleaningConfig):
        self.config = config

    def clean(self):
        logger.info("Reading raw data...")
        data = pd.read_csv(self.config.data_path)

        df = data[(data['condition'] == 'Birth Control') | (data['condition'] == 'Depression') | (data['condition'] == 'Pain') | (data['condition'] == 'Anxiety') | (data['condition'] == 'Acne') | (data['condition'] == 'Diabetes, Type 2') | (data['condition'] == 'High Blood Pressure')]

        if 'review' not in df.columns or 'condition' not in df.columns:
            raise ValueError("Input data must contain 'review' and 'condition' columns.")

        logger.info("Cleaning review texts...")
        df['clean_review'] = df['review'].astype(str).apply(clean_text)

        df.dropna(subset=['clean_review', 'condition'], inplace=True)

        create_directories([os.path.dirname(self.config.cleaned_data_path)])
        df.to_csv(self.config.cleaned_data_path, index=False, na_rep="")

        logger.info(f"Cleaned data saved at: {self.config.cleaned_data_path}")