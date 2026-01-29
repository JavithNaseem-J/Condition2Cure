"""
Utility Functions
=================
Simple, focused helper functions.
No unnecessary decorators or over-engineering.
"""
import os
import json
import yaml
import joblib
from pathlib import Path
from typing import Any, Dict
from Condition2Cure import logger


def read_yaml(path: str) -> Dict:
    """Read YAML file and return as dictionary."""
    with open(path, "r") as f:
        content = yaml.safe_load(f)
    logger.info(f"Loaded YAML: {path}")
    return content


def save_json(path: str, data: Dict) -> None:
    """Save dictionary to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved JSON: {path}")


def load_json(path: str) -> Dict:
    """Load JSON file as dictionary."""
    with open(path, "r") as f:
        content = json.load(f)
    logger.info(f"Loaded JSON: {path}")
    return content


def save_model(model: Any, path: str) -> None:
    """Save model using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Saved model: {path}")


def load_model(path: str) -> Any:
    """Load model using joblib."""
    model = joblib.load(path)
    logger.info(f"Loaded model: {path}")
    return model


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def get_file_size(path: str) -> str:
    """Get file size in human-readable format."""
    size_bytes = os.path.getsize(path)
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"