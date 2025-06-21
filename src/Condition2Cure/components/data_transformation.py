import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from Condition2Cure.utils.helpers import create_directories
from Condition2Cure import logger
from Condition2Cure.entities.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform(self):
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
        X_tfidf = vectorizer.fit_transform(df['clean_review'])

        logger.info("Reducing dimensionality using TruncatedSVD...")
        svd = TruncatedSVD(n_components=self.config.svd_components)
        X_reduced = svd.fit_transform(X_tfidf)

        logger.info("Encoding labels...")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[self.config.target_column])

        logger.info("Saving vectorizer, SVD, label encoder...")
        create_directories([os.path.dirname(self.config.vectorizer_path)])
        joblib.dump(vectorizer, self.config.vectorizer_path)
        joblib.dump(svd, self.config.svd_path)
        joblib.dump(label_encoder, self.config.label_encoder_path)

        logger.info("Saving features and labels...")
        np.save(self.config.features_path, X_reduced)
        np.save(self.config.labels_path, y)

        logger.info("Data transformation complete.")