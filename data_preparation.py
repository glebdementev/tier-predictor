"""
Data preparation utilities for forest element tier prediction.

This module handles loading, cleaning, and preparing data for model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from features import extract_composition_features, get_element_share, is_element_dominant


def load_data(csv_path: str = 'combined_data.csv') -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        DataFrame with the loaded data
    """
    return pd.read_csv(csv_path)


def prepare_data(df: pd.DataFrame) -> tuple:
    """
    Prepare data for training by extracting and engineering features.
    
    NOTE: Diameter is NOT used in this model - only height.
    
    Args:
        df: Input DataFrame with raw data
    
    Returns:
        Tuple of (X, y, feature_names, label_encoder, cleaned_data)
    """
    # Work with a copy
    data = df.copy()
    
    # Remove rows where target (Разряд) is missing
    data = data.dropna(subset=['Разряд'])
    
    # Remove rows where Высота is missing (we only need height, not diameter)
    data = data.dropna(subset=['Высота'])
    
    # Convert target to integer
    data['Разряд'] = data['Разряд'].astype(int)
    
    # Feature engineering
    features = []
    
    for idx, row in data.iterrows():
        feat = {}
        
        # Basic numerical features (only height, NO diameter)
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
        
        # Is element dominant (has highest share)?
        feat['is_dominant'] = is_element_dominant(comp_features, element)
        
        features.append(feat)
    
    # Create feature DataFrame
    feature_df = pd.DataFrame(features)
    
    # Encode categorical variable (element_type)
    le = LabelEncoder()
    feature_df['element_type_encoded'] = le.fit_transform(feature_df['element_type'])
    
    # Select numerical features for training (WITHOUT diameter)
    feature_columns = [
        'height',
        'num_species', 
        'total_composition', 
        'element_share', 
        'is_dominant', 
        'element_type_encoded'
    ]
    
    X = feature_df[feature_columns].values
    y = data['Разряд'].values
    
    return X, y, feature_columns, le, data


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics of the dataset.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dictionary with summary statistics
    """
    return {
        'total_rows': len(df),
        'rows_with_tier': df['Разряд'].notna().sum(),
        'rows_with_height': df['Высота'].notna().sum(),
        'unique_elements': df['Элемент леса'].nunique(),
        'unique_tiers': sorted(df['Разряд'].dropna().unique().astype(int).tolist()),
    }

