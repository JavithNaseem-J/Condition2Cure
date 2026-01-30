import os
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from Condition2Cure import logger
from Condition2Cure.config import config
from Condition2Cure.utils.nlp_utils import get_embeddings_batch


def transform_data(input_path: str = None) -> dict:

    input_path = input_path or config.cleaned_data_path
    
    logger.info(f"Loading cleaned data: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} records")
    
    # Get text column
    text_col = 'clean_review' if 'clean_review' in df.columns else config.text_column
    df[text_col] = df[text_col].fillna("").astype(str)
    
    # Remove empty reviews
    df = df[df[text_col].str.len() > 0]
    logger.info(f"Records after removing empty: {len(df)}")
    
    # Generate BERT embeddings
    logger.info("Generating BERT embeddings (this may take a few minutes)...")
    texts = df[text_col].tolist()
    X = get_embeddings_batch(texts)
    logger.info(f"Embeddings shape: {X.shape}")  # Should be (n_samples, 384)
    
    # Encode target labels
    logger.info("Encoding condition labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[config.target_column])
    
    n_classes = len(label_encoder.classes_)
    logger.info(f"Found {n_classes} unique conditions: {list(label_encoder.classes_)}")
    

    logger.info(f"Splitting data: {1-config.test_size:.0%} train, {config.test_size:.0%} test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples (held out for final evaluation)")
    
    # Save all artifacts
    features_dir = os.path.dirname(config.features_path)
    os.makedirs(features_dir, exist_ok=True)
    
    # Save train set
    np.save(config.features_path, X_train)
    np.save(config.labels_path, y_train)
    
    # Save test set (for evaluation ONLY)
    np.save(config.test_features_path, X_test)
    np.save(config.test_labels_path, y_test)
    
    # Save label encoder
    joblib.dump(label_encoder, config.label_encoder_path)
    
    logger.info("Data transformation complete!")
    logger.info(f"Train features: {config.features_path}")
    logger.info(f"Test features: {config.test_features_path}")
    
    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X.shape[1],
        "n_classes": n_classes,
        "classes": list(label_encoder.classes_)
    }



if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info(">>>>>> Stage: Data Transformation started <<<<<<")
    logger.info("=" * 60)
    
    info = transform_data()
    
    logger.info(f"Train samples: {info['n_train']}")
    logger.info(f"Test samples: {info['n_test']} (held out)")
    logger.info(f"Classes: {info['n_classes']}")
    logger.info(">>>>>> Stage: Data Transformation completed <<<<<<")
