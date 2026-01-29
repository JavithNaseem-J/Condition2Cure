"""
Prediction Pipeline
===================
Handles real-time predictions from user input.

Flow:
1. User enters symptom description
2. Text is converted to BERT embedding
3. XGBoost predicts the condition
4. Label encoder converts number back to condition name
"""
import joblib
import numpy as np
from Condition2Cure.utils.nlp_utils import get_embedding


class PredictionPipeline:
    """
    Simple prediction pipeline for medical condition classification.
    
    Usage:
        pipeline = PredictionPipeline()
        condition, confidence = pipeline.predict("I have a headache...")
    """
    
    def __init__(self):
        """Load the trained model and label encoder."""
        # Load model (trained XGBoost)
        self.model = joblib.load("artifacts/model_training/model.joblib")
        
        # Load label encoder (to convert numbers back to condition names)
        self.label_encoder = joblib.load("artifacts/feature_engineering/label_encoder.pkl")
        
        print(f"Model loaded! Can predict {len(self.label_encoder.classes_)} conditions")

    def predict(self, text: str):
        """
        Predict medical condition from symptom description.
        
        Args:
            text: Patient's symptom description
            
        Returns:
            tuple: (condition_name, confidence_score)
            
        Example:
            >>> pipeline = PredictionPipeline()
            >>> condition, confidence = pipeline.predict("severe headache and nausea")
            >>> print(f"{condition} ({confidence:.1%})")
            Migraine (87.3%)
        """
        if not text or len(text.strip()) < 5:
            raise ValueError("Please provide a valid description")
        
        # Step 1: Convert text to BERT embedding
        embedding = get_embedding(text)
        
        # Step 2: Reshape for prediction (model expects 2D array)
        X = embedding.reshape(1, -1)
        
        # Step 3: Get prediction
        pred_class = self.model.predict(X)[0]
        
        # Step 4: Get confidence (probability of predicted class)
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            confidence = float(proba[pred_class])
        else:
            confidence = 1.0
        
        # Step 5: Convert class number to condition name
        condition = self.label_encoder.inverse_transform([pred_class])[0]
        
        return condition, confidence

    def predict_top_k(self, text: str, k: int = 3):
        """
        Get top K most likely conditions.
        
        Useful when confidence is low to show alternatives.
        
        Args:
            text: Symptom description
            k: Number of top predictions
            
        Returns:
            list of (condition, confidence) tuples
        """
        embedding = get_embedding(text)
        X = embedding.reshape(1, -1)
        
        # Get probabilities for all classes
        proba = self.model.predict_proba(X)[0]
        
        # Get top K indices
        top_indices = np.argsort(proba)[::-1][:k]
        
        # Build results
        results = []
        for idx in top_indices:
            condition = self.label_encoder.inverse_transform([idx])[0]
            confidence = float(proba[idx])
            results.append((condition, confidence))
        
        return results
