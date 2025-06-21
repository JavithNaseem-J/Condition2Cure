import joblib
import pandas as pd
from pathlib import Path
from Condition2Cure.utils.nlp_utils import clean_text

class PredictionPipeline:
    def __init__(self):
        self.vectorizer = joblib.load(Path("artifacts/feature_engineering/vectorizer.pkl"))
        self.model = joblib.load(Path("artifacts/model_training/model.joblib"))
        self.label_encoder = joblib.load(Path("artifacts/feature_engineering/label_encoder.pkl"))

    def predict(self, raw_text: str) -> str:
        clean = clean_text(raw_text)
        vec = self.vectorizer.transform([clean])
        pred_class = self.model.predict(vec)[0]
        return self.label_encoder.inverse_transform([pred_class])[0]
