import pickle
import pandas as pd
import numpy as np
import mlflow
from typing import Dict, Any
from ..utils.config_manager import config

class ModelPredictor:
    def __init__(self, model_path: str = None):
        if model_path:
            self.model = self.load_model(model_path)
        else:
            self.model = None
    
    def load_model(self, model_path: str):
        """Charge un modèle sauvegardé"""
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fait des prédictions sur de nouvelles données"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")
        
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)[:, 1]
        
        return pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities
        })
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Retourne les probabilités de prédiction"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")
        
        return self.model.predict_proba(features)