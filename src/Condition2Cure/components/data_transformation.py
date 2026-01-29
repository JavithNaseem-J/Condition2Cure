import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from Condition2Cure.utils.nlp_utils import get_embeddings_batch
from Condition2Cure import logger


class DataTransformation:
  
    def __init__(self, config):

        self.config = config

    def transform(self):

        logger.info("Loading cleaned data...")
        df = pd.read_csv(self.config.cleaned_data_path)
        logger.info(f"Loaded {len(df)} records")

        text_col = 'clean_review' if 'clean_review' in df.columns else 'review'
        df[text_col] = df[text_col].fillna("").astype(str)
        
        # Remove empty reviews
        df = df[df[text_col].str.len() > 0]
        logger.info(f"Records after removing empty: {len(df)}")

        # Step 3: Generate BERT embeddings
        # This is the key innovation - using pre-trained BERT instead of TF-IDF
        logger.info("Generating BERT embeddings (this may take a few minutes)...")
        texts = df[text_col].tolist()
        X = get_embeddings_batch(texts)
        logger.info(f"Embeddings shape: {X.shape}")  # Should be (n_samples, 384)

        # Step 4: Encode target labels
        logger.info("Encoding condition labels...")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[self.config.target_column])
        
        n_classes = len(label_encoder.classes_)
        logger.info(f"Found {n_classes} unique conditions")

        os.makedirs(os.path.dirname(self.config.features_path), exist_ok=True)
        
        np.save(self.config.features_path, X)
        np.save(self.config.labels_path, y)
        
        joblib.dump(label_encoder, self.config.label_encoder_path)

        logger.info("Data transformation complete!")
        logger.info(f"Features saved to: {self.config.features_path}")
        
        return {
            "n_samples": len(df),
            "n_features": 384,
            "n_classes": n_classes
        }
