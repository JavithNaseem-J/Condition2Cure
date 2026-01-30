import joblib
import numpy as np
from Condition2Cure.config import config
from Condition2Cure.utils.nlp_utils import get_embedding


class PredictionPipeline:

    
    def __init__(self):
        """Load the trained model and label encoder."""
        self.model = joblib.load(config.model_path)
        self.label_encoder = joblib.load(config.label_encoder_path)
        print(f"Model loaded! Can predict {len(self.label_encoder.classes_)} conditions")

    def predict(self, text: str) -> tuple:

        if not text or len(text.strip()) < 5:
            raise ValueError("Please provide a valid description (at least 5 characters)")
        
        # Convert text to BERT embedding
        embedding = get_embedding(text)
        X = embedding.reshape(1, -1)
        
        # Get prediction
        pred_class = self.model.predict(X)[0]
        
        # Get confidence
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            confidence = float(proba[pred_class])
        else:
            confidence = 1.0
        
        # Convert to condition name
        condition = self.label_encoder.inverse_transform([pred_class])[0]
        
        return condition, confidence

    def predict_top_k(self, text: str, k: int = 3) -> list:

        embedding = get_embedding(text)
        X = embedding.reshape(1, -1)
        
        proba = self.model.predict_proba(X)[0]
        top_indices = np.argsort(proba)[::-1][:k]
        
        results = []
        for idx in top_indices:
            condition = self.label_encoder.inverse_transform([idx])[0]
            confidence = float(proba[idx])
            results.append((condition, confidence))
        
        return results
