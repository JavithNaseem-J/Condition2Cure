from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    
    # === Paths ===
    artifacts_root: str = "artifacts"
    
    # Data paths
    data_source_id: str = "10dmW5TSCk91lukqWcAeBSt9pjjv-xaFW"
    raw_data_path: str = "artifacts/data_ingestion/Drugs_Data.csv"
    cleaned_data_path: str = "artifacts/data_cleaning/cleaned.csv"
    
    # Feature paths
    features_path: str = "artifacts/feature_engineering/X.npy"
    labels_path: str = "artifacts/feature_engineering/y.npy"
    test_features_path: str = "artifacts/feature_engineering/X_test.npy"
    test_labels_path: str = "artifacts/feature_engineering/y_test.npy"
    label_encoder_path: str = "artifacts/feature_engineering/label_encoder.pkl"
    
    # Model paths
    model_path: str = "artifacts/model_training/model.joblib"
    metrics_path: str = "artifacts/model/metrics.json"
    confusion_matrix_path: str = "artifacts/model/confusion_matrix.png"
    
    # === Training Parameters ===
    test_size: float = 0.2
    random_state: int = 42
    optuna_trials: int = 20
    cv_folds: int = 3
    
    # Debug: Limit data size for faster testing (set to None for full training)
    debug_sample_size: int = 2000
    
    # === Data Parameters ===
    target_column: str = "condition"
    text_column: str = "review"
    
    conditions: List[str] = field(default_factory=lambda: [
        "Birth Control",
        "Depression",
        "Pain",
        "Anxiety",
        "Acne",
        "Diabetes, Type 2",
        "High Blood Pressure"
    ])
    
    required_columns: List[str] = field(default_factory=lambda: [
        "drugName", "condition", "review", "rating", "usefulCount"
    ])


# Single global config instance
config = Config()
