"""
Configuration Entities
======================
Dataclasses to hold configuration for each pipeline stage.
Simple and easy to understand.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    """Config for downloading and extracting data."""
    root_dir: Path
    source_id: str
    local_data_file: Path
    unzip_dir: Path


@dataclass
class DataValidationConfig:
    """Config for validating data schema."""
    root_dir: Path
    status_file: str
    unzip_data_dir: Path
    all_schema: dict


@dataclass
class DataCleaningConfig:
    """Config for cleaning raw data."""
    root_dir: Path
    data_path: Path
    cleaned_data_path: Path


@dataclass
class DataTransformationConfig:
    """Config for feature engineering."""
    root_dir: Path
    cleaned_data_path: str
    target_column: str
    features_path: str
    labels_path: str
    label_encoder_path: str
    # Legacy fields (not used with BERT, kept for compatibility)
    max_features: int = 5000
    ngram_range: tuple = (1, 2)
    svd_components: int = 300
    vectorizer_path: str = ""
    svd_path: str = ""


@dataclass
class ModelTrainerConfig:
    """Config for model training."""
    root_dir: Path
    features_path: Path
    labels_path: Path
    model_path: Path
    max_iter: int = 1000
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class ModelEvaluationConfig:
    """Config for model evaluation."""
    root_dir: Path
    features_path: Path
    labels_path: Path
    model_path: Path
    label_encoder_path: Path
    test_size: float
    random_state: int
    metrics_path: Path
    cm_path: Path


@dataclass
class ModelRegistryConfig:
    """Config for MLflow model registry."""
    model_name: str
    metric_path: Path
    metric_key: str