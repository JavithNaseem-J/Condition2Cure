from dataclasses import dataclass
from pathlib import Path



@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_id:str
    local_data_file:Path
    unzip_dir:Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    status_file: str
    unzip_data_dir: Path
    all_schema: dict

@dataclass(frozen=True)
class DataCleaningConfig:
    root_dir: Path
    data_path: Path
    cleaned_data_path: Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    cleaned_data_path: str
    target_column: str
    max_features: int
    ngram_range: tuple
    svd_components: int
    features_path: str
    labels_path: str
    vectorizer_path: str
    svd_path: str
    label_encoder_path: str

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    features_path: Path
    labels_path: Path
    model_path: Path
    max_iter: int
    test_size: float
    random_state: int
    

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    features_path: Path
    labels_path: Path
    model_path: Path
    label_encoder_path: Path
    test_size: float
    random_state: int
    metrics_path: Path
    cm_path: Path


@dataclass(frozen=True)
class ModelRegistryConfig:
    model_name: str
    metric_path: Path
    metric_key: str


@dataclass(frozen=True)
class DriftMonitoringConfig:
    reference_data: Path
    current_data: Path
    output_dir: Path