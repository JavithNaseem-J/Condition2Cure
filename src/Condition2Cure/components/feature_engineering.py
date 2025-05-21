import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from Condition2Cure.entities.config_entity import FeatureEngineeringConfig
from Condition2Cure import logger

class FeatureEngineering:
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    def transfom(self):
        logger.info("Loading cleaned data...")
        df = pd.read_csv(self.config.cleaned_data_path)

        if 'clean_review' not in df.columns or self.config.target_column not in df.columns:
            raise ValueError("Required columns missing in cleaned data.")

        df['clean_review'] = df['clean_review'].fillna("")


        logger.info("Fitting TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=tuple(self.config.ngram_range)
        )
        X = vectorizer.fit_transform(df['clean_review'])

        logger.info("Encoding target labels...")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[self.config.target_column])

        logger.info(f"TF-IDF features shape: {X.shape}")
        logger.info(f"Number of classes: {len(np.unique(y))}")

        logger.info("Saving vectorizer and label encoder...")
        os.makedirs(os.path.dirname(self.config.vectorizer_path), exist_ok=True)
        joblib.dump(vectorizer, self.config.vectorizer_path)
        joblib.dump(label_encoder, self.config.label_encoder_path)

        logger.info("Saving feature matrix and labels...")
        os.makedirs(os.path.dirname(self.config.features_path), exist_ok=True)
        np.save(self.config.features_path, {'X': X, 'y': y})

        logger.info("Feature engineering completed.")
