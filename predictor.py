"""
TierPredictor class for forest element tier prediction.

This module provides a high-level interface for training and using the tier prediction model.
"""

import numpy as np
import pandas as pd

from features import extract_composition_features, get_element_share, is_element_dominant
from data_preparation import load_data, prepare_data, get_data_summary
from models import train_and_evaluate_models, save_model, load_model


class TierPredictor:
    """
    A class to predict the tier (разряд) of a forest element.
    
    This predictor uses only composition, element type, and height.
    Diameter is NOT used.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.model_name = None
        self.is_trained = False
    
    def train(self, csv_path: str = 'combined_data.csv') -> 'TierPredictor':
        """
        Train the model on the dataset.
        
        Args:
            csv_path: Path to the CSV data file
        
        Returns:
            self for method chaining
        """
        print("Loading data...")
        df = load_data(csv_path)
        
        summary = get_data_summary(df)
        print(f"Total rows: {summary['total_rows']}")
        print(f"Rows with tier (Разряд): {summary['rows_with_tier']}")
        print(f"Unique tiers: {summary['unique_tiers']}")
        
        print("\nPreparing features...")
        X, y, feature_names, le, data = prepare_data(df)
        self.label_encoder = le
        self.feature_names = feature_names
        
        print(f"Training samples: {len(X)}")
        print(f"Features used: {feature_names}")
        print(f"NOTE: Diameter is NOT used - only height and composition features")
        
        self.model, self.scaler, self.model_name, _ = train_and_evaluate_models(X, y, feature_names)
        self.is_trained = True
        
        return self
    
    def save(self, save_path: str = 'tier_predictor_model.joblib') -> str:
        """
        Save the trained model to disk.
        
        Args:
            save_path: Path to save the model
        
        Returns:
            Path where model was saved
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first.")
        
        return save_model(
            self.model, 
            self.scaler, 
            self.label_encoder, 
            self.feature_names,
            self.model_name,
            save_path
        )
    
    def load(self, load_path: str = 'tier_predictor_model.joblib') -> 'TierPredictor':
        """
        Load a trained model from disk.
        
        Args:
            load_path: Path to the saved model
        
        Returns:
            self for method chaining
        """
        model_data = load_model(load_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.model_name = model_data['model_name']
        self.is_trained = True
        
        print(f"Model loaded: {self.model_name}")
        
        return self
    
    def _extract_features(self, composition: str, element: str, height: float) -> np.ndarray:
        """
        Extract features from input parameters.
        
        Args:
            composition: The composition string (e.g., "5С2Б2К1Е")
            element: The forest element type (e.g., "С")
            height: The height of the element
        
        Returns:
            Feature array
        """
        feat = {}
        
        feat['height'] = height if height and height > 0 else 0
        
        # Composition features
        comp_features = extract_composition_features(composition)
        feat['num_species'] = comp_features.get('num_species', 0)
        feat['total_composition'] = comp_features.get('total_composition', 0)
        
        # Element share
        element_upper = element.upper() if element else ''
        feat['element_share'] = get_element_share(comp_features, element_upper, feat['total_composition'])
        
        # Is dominant
        feat['is_dominant'] = is_element_dominant(comp_features, element_upper)
        
        # Encode element type
        try:
            feat['element_type_encoded'] = self.label_encoder.transform([element])[0]
        except ValueError:
            # Unknown element type - use mode of encoded values
            feat['element_type_encoded'] = 0
        
        # Create feature array (order must match training)
        X = np.array([[
            feat['height'],
            feat['num_species'], 
            feat['total_composition'], 
            feat['element_share'],
            feat['is_dominant'], 
            feat['element_type_encoded']
        ]])
        
        return X
    
    def predict(self, composition: str, element: str, height: float) -> int:
        """
        Predict the tier for a single forest element.
        
        NOTE: Does NOT use diameter - only height.
        
        Args:
            composition: The composition string (e.g., "5С2Б2К1Е")
            element: The forest element type (e.g., "С", "Б", "Л")
            height: The height of the element
        
        Returns:
            Predicted tier (разряд)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() or load() first.")
        
        X = self._extract_features(composition, element, height)
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        
        return int(prediction)
    
    def predict_proba(self, composition: str, element: str, height: float) -> dict:
        """
        Predict the tier probabilities for a single forest element.
        
        NOTE: Does NOT use diameter - only height.
        
        Args:
            composition: The composition string (e.g., "5С2Б2К1Е")
            element: The forest element type (e.g., "С")
            height: The height of the element
        
        Returns:
            Dictionary mapping tier to probability
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() or load() first.")
        
        X = self._extract_features(composition, element, height)
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(X_scaled)[0]
            classes = self.model.classes_
            return {int(c): float(p) for c, p in zip(classes, probas)}
        else:
            prediction = self.model.predict(X_scaled)[0]
            return {int(prediction): 1.0}
    
    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict tiers for multiple forest elements.
        
        Args:
            data: DataFrame with columns 'Состав', 'Элемент леса', 'Высота'
        
        Returns:
            DataFrame with added 'predicted_tier' column
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() or load() first.")
        
        result = data.copy()
        predictions = []
        
        for _, row in data.iterrows():
            pred = self.predict(
                composition=row['Состав'],
                element=row['Элемент леса'],
                height=row['Высота']
            )
            predictions.append(pred)
        
        result['predicted_tier'] = predictions
        return result

