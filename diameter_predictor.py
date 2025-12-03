"""
DiameterPredictor class for forest element diameter prediction.

This module provides a high-level interface for training and using the diameter prediction model.
Predicts diameter using: composition, element type, and height.
Does NOT use tier (Разряд).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from features import extract_composition_features, get_element_share, is_element_dominant
from regression_models import (
    train_and_evaluate_regression_models, 
    save_regression_model, 
    load_regression_model
)


def prepare_diameter_data(df: pd.DataFrame) -> tuple:
    """
    Prepare data for diameter prediction training.
    
    Uses: Состав, Элемент леса, Высота
    Does NOT use: Разряд (tier)
    
    Args:
        df: Input DataFrame with raw data
    
    Returns:
        Tuple of (X, y, feature_names, label_encoder, cleaned_data)
    """
    # Work with a copy
    data = df.copy()
    
    # Remove rows where target (Диаметр) is missing
    data = data.dropna(subset=['Диаметр'])
    
    # Remove rows where Высота is missing
    data = data.dropna(subset=['Высота'])
    
    # Feature engineering
    features = []
    
    for idx, row in data.iterrows():
        feat = {}
        
        # Basic numerical features (height only, NO tier)
        feat['height'] = row['Высота'] if pd.notna(row['Высота']) else 0
        
        # Tree element type
        feat['element_type'] = str(row['Элемент леса']) if pd.notna(row['Элемент леса']) else 'UNKNOWN'
        
        # Composition features
        comp_features = extract_composition_features(row['Состав'])
        feat['num_species'] = comp_features.get('num_species', 0)
        feat['total_composition'] = comp_features.get('total_composition', 0)
        
        # Get the share of current element in composition
        element = str(row['Элемент леса']).upper() if pd.notna(row['Элемент леса']) else ''
        feat['element_share'] = get_element_share(comp_features, element, feat['total_composition'])
        
        # Is element dominant?
        feat['is_dominant'] = is_element_dominant(comp_features, element)
        
        features.append(feat)
    
    # Create feature DataFrame
    feature_df = pd.DataFrame(features)
    
    # Encode categorical variable (element_type)
    le = LabelEncoder()
    feature_df['element_type_encoded'] = le.fit_transform(feature_df['element_type'])
    
    # Select numerical features for training (NO tier)
    feature_columns = [
        'height',
        'num_species', 
        'total_composition', 
        'element_share', 
        'is_dominant', 
        'element_type_encoded'
    ]
    
    X = feature_df[feature_columns].values
    y = data['Диаметр'].values
    
    return X, y, feature_columns, le, data


class DiameterPredictor:
    """
    A class to predict the diameter (диаметр) of a forest element.
    
    Uses: composition, element type, height
    Does NOT use: tier (Разряд)
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.model_name = None
        self.is_trained = False
    
    def train(self, csv_path: str = 'combined_data.csv') -> 'DiameterPredictor':
        """
        Train the diameter prediction model on the dataset.
        
        Args:
            csv_path: Path to the CSV data file
        
        Returns:
            self for method chaining
        """
        print("Loading data...")
        df = pd.read_csv(csv_path)
        
        print(f"Total rows: {len(df)}")
        print(f"Rows with diameter (Диаметр): {df['Диаметр'].notna().sum()}")
        print(f"Rows with height (Высота): {df['Высота'].notna().sum()}")
        
        print("\nPreparing features for diameter prediction...")
        X, y, feature_names, le, data = prepare_diameter_data(df)
        self.label_encoder = le
        self.feature_names = feature_names
        
        print(f"Training samples: {len(X)}")
        print(f"Features used: {feature_names}")
        print(f"NOTE: Tier (Разряд) is NOT used - only height, composition, element")
        print(f"\nDiameter statistics:")
        print(f"  Min: {y.min():.2f}")
        print(f"  Max: {y.max():.2f}")
        print(f"  Mean: {y.mean():.2f}")
        print(f"  Std: {y.std():.2f}")
        
        self.model, self.scaler, self.model_name, _ = train_and_evaluate_regression_models(
            X, y, feature_names
        )
        self.is_trained = True
        
        return self
    
    def save(self, save_path: str = 'diameter_predictor_model.joblib') -> str:
        """
        Save the trained model to disk.
        
        Args:
            save_path: Path to save the model
        
        Returns:
            Path where model was saved
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first.")
        
        return save_regression_model(
            self.model, 
            self.scaler, 
            self.label_encoder, 
            self.feature_names,
            self.model_name,
            save_path
        )
    
    def load(self, load_path: str = 'diameter_predictor_model.joblib') -> 'DiameterPredictor':
        """
        Load a trained model from disk.
        
        Args:
            load_path: Path to the saved model
        
        Returns:
            self for method chaining
        """
        model_data = load_regression_model(load_path)
        
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
    
    def predict(self, composition: str, element: str, height: float) -> float:
        """
        Predict the diameter for a single forest element.
        
        NOTE: Does NOT use tier - only height, composition, element.
        
        Args:
            composition: The composition string (e.g., "5С2Б2К1Е")
            element: The forest element type (e.g., "С", "Б", "Л")
            height: The height of the element
        
        Returns:
            Predicted diameter (диаметр)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() or load() first.")
        
        X = self._extract_features(composition, element, height)
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        
        return float(prediction)
    
    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict diameters for multiple forest elements.
        
        Args:
            data: DataFrame with columns 'Состав', 'Элемент леса', 'Высота'
        
        Returns:
            DataFrame with added 'predicted_diameter' column
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
        
        result['predicted_diameter'] = predictions
        return result

