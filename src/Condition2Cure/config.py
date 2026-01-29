"""
Condition2Cure - Unified Configuration
=======================================
Single source of truth for all configuration.
No more 5 layers of indirection!
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    """All configuration in one place."""
    
    # === Paths ===
    artifacts_root: str = "artifacts"
    
    # Data paths
    data_source_id: str = "10dmW5TSCk91lukqWcAeBSt9pjjv-xaFW"
    raw_data_path: str = "artifacts/data_ingestion/Drugs_Data.csv"
    cleaned_data_path: str = "artifacts/data_cleaning/cleaned.csv"
    
    # Feature paths
    features_path: str = "artifacts/features/X_train.npy"
    labels_path: str = "artifacts/features/y_train.npy"
    test_features_path: str = "artifacts/features/X_test.npy"
    test_labels_path: str = "artifacts/features/y_test.npy"
    label_encoder_path: str = "artifacts/features/label_encoder.pkl"
    
    # Model paths
    model_path: str = "artifacts/model/model.joblib"
    metrics_path: str = "artifacts/model/metrics.json"
    confusion_matrix_path: str = "artifacts/model/confusion_matrix.png"
    
    # === Training Parameters ===
    test_size: float = 0.2
    random_state: int = 42
    optuna_trials: int = 20
    cv_folds: int = 3
    
    # === Data Parameters ===
    target_column: str = "condition"
    text_column: str = "review"
    
    # Conditions to include (moved from hardcoded filter)
    conditions: List[str] = field(default_factory=lambda: [
        "Birth Control",
        "Depression",
        "Pain",
        "Anxiety",
        "Acne",
        "Diabetes, Type 2",
        "High Blood Pressure"
    ])
    
    # === Schema (for validation) ===
    required_columns: List[str] = field(default_factory=lambda: [
        "drugName", "condition", "review", "rating", "usefulCount"
    ])


# Single global config instance
config = Config()
