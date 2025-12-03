"""
Feature extraction utilities for forest element tier prediction.

This module contains functions to extract features from forest composition data.
"""

import re
import pandas as pd


def extract_composition_features(composition: str) -> dict:
    """
    Extract features from the composition string (Состав).
    
    The composition format is like "5С2Б2К1Е" which means:
    - 5 parts of С (сосна/pine)
    - 2 parts of Б (береза/birch)
    - 2 parts of К (кедр/cedar)
    - 1 part of Е (ель/spruce)
    
    Args:
        composition: The composition string (e.g., "5С2Б2К1Е")
    
    Returns:
        Dictionary with counts for each tree type and derived features.
    """
    if pd.isna(composition):
        return {}
    
    # Pattern to match number followed by letters (tree type)
    pattern = r'(\d+)([А-ЯЁа-яё]+)'
    matches = re.findall(pattern, str(composition))
    
    features = {}
    total = 0
    for count, tree_type in matches:
        count = int(count)
        # Normalize tree type names (handle variations)
        tree_type = tree_type.upper()
        if tree_type in features:
            features[tree_type] += count
        else:
            features[tree_type] = count
        total += count
    
    features['total_composition'] = total
    features['num_species'] = len([k for k in features.keys() if k != 'total_composition'])
    
    return features


def get_element_share(composition_features: dict, element: str, total_composition: int) -> float:
    """
    Calculate the share of a specific element in the composition.
    
    Args:
        composition_features: Dictionary of composition features
        element: The element type (e.g., "С")
        total_composition: Total composition value
    
    Returns:
        Share of the element (0.0 to 1.0)
    """
    if not element or total_composition <= 0:
        return 0.0
    
    element_upper = element.upper()
    return composition_features.get(element_upper, 0) / total_composition


def is_element_dominant(composition_features: dict, element: str) -> int:
    """
    Check if the given element is the dominant one in the composition.
    
    Args:
        composition_features: Dictionary of composition features
        element: The element type (e.g., "С")
    
    Returns:
        1 if element is dominant, 0 otherwise
    """
    if not element or not composition_features:
        return 0
    
    element_upper = element.upper()
    tree_types = [k for k in composition_features.keys() 
                  if k not in ['total_composition', 'num_species']]
    
    if not tree_types:
        return 0
    
    max_share = max([composition_features.get(t, 0) for t in tree_types], default=0)
    return 1 if composition_features.get(element_upper, 0) == max_share else 0

