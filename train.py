"""
Tier (Разряд) and Diameter (Диаметр) Prediction Models for Forest Elements

This script trains machine learning models to predict:
1. Tier (разряд) - classification using: composition, element type, height
2. Diameter (диаметр) - regression using: composition, element type, height

NOTE: 
- Tier prediction does NOT use diameter
- Diameter prediction does NOT use tier

Usage:
    python train.py                          # Train both models
    python train.py --tier-only              # Train tier model only
    python train.py --diameter-only          # Train diameter model only
    python train.py --data path/to/data.csv  # Custom data file
"""

import argparse
from predictor import TierPredictor
from diameter_predictor import DiameterPredictor


def train_tier_model(data_path: str, output_path: str):
    """Train the tier prediction model."""
    print("\n" + "=" * 80)
    print("TRAINING TIER (РАЗРЯД) PREDICTION MODEL")
    print("Using: composition, element type, height")
    print("NOT using: diameter")
    print("=" * 80)
    
    predictor = TierPredictor()
    predictor.train(data_path)
    predictor.save(output_path)
    
    # Demonstrate predictions
    print("\n" + "-" * 60)
    print("TIER PREDICTION EXAMPLES")
    print("-" * 60)
    
    examples = [
        ("5С2Б2К1Е", "С", 12.0),
        ("5С2Б2К1Е", "Б", 10.0),
        ("5С2Б2К1Е", "К", 14.0),
        ("6Л2К1Е1Б+С", "Л", 19.0),
        ("7Б3Л", "Б", 2.0),
    ]
    
    print(f"\n{'Состав':<20} {'Элемент':<10} {'Высота':<10} {'Tier':>8}")
    print("-" * 55)
    
    for composition, element, height in examples:
        tier = predictor.predict(composition, element, height)
        print(f"{composition:<20} {element:<10} {height:<10.1f} {tier:>8}")
    
    return predictor


def train_diameter_model(data_path: str, output_path: str):
    """Train the diameter prediction model."""
    print("\n" + "=" * 80)
    print("TRAINING DIAMETER (ДИАМЕТР) PREDICTION MODEL")
    print("Using: composition, element type, height")
    print("NOT using: tier")
    print("=" * 80)
    
    predictor = DiameterPredictor()
    predictor.train(data_path)
    predictor.save(output_path)
    
    # Demonstrate predictions
    print("\n" + "-" * 60)
    print("DIAMETER PREDICTION EXAMPLES")
    print("-" * 60)
    
    examples = [
        ("5С2Б2К1Е", "С", 12.0),
        ("5С2Б2К1Е", "Б", 10.0),
        ("5С2Б2К1Е", "К", 14.0),
        ("6Л2К1Е1Б+С", "Л", 19.0),
        ("7Б3Л", "Б", 2.0),
    ]
    
    print(f"\n{'Состав':<20} {'Элемент':<10} {'Высота':<10} {'Diameter':>10}")
    print("-" * 55)
    
    for composition, element, height in examples:
        diameter = predictor.predict(composition, element, height)
        print(f"{composition:<20} {element:<10} {height:<10.1f} {diameter:>10.2f}")
    
    return predictor


def main():
    """Main function to train and save prediction models."""
    parser = argparse.ArgumentParser(
        description='Train tier and diameter prediction models for forest elements'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default='combined_data.csv',
        help='Path to the training data CSV file (default: combined_data.csv)'
    )
    parser.add_argument(
        '--tier-output', 
        type=str, 
        default='tier_predictor_model.joblib',
        help='Path to save the tier model (default: tier_predictor_model.joblib)'
    )
    parser.add_argument(
        '--diameter-output', 
        type=str, 
        default='diameter_predictor_model.joblib',
        help='Path to save the diameter model (default: diameter_predictor_model.joblib)'
    )
    parser.add_argument(
        '--tier-only',
        action='store_true',
        help='Train only the tier prediction model'
    )
    parser.add_argument(
        '--diameter-only',
        action='store_true',
        help='Train only the diameter prediction model'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("FOREST ELEMENT PREDICTOR - TRAINING")
    print("=" * 80)
    
    tier_predictor = None
    diameter_predictor = None
    
    # Train tier model
    if not args.diameter_only:
        tier_predictor = train_tier_model(args.data, args.tier_output)
    
    # Train diameter model
    if not args.tier_only:
        diameter_predictor = train_diameter_model(args.data, args.diameter_output)
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 80)
    
    if tier_predictor:
        print(f"\nTier Model:")
        print(f"  Best model: {tier_predictor.model_name}")
        print(f"  Saved to: {args.tier_output}")
    
    if diameter_predictor:
        print(f"\nDiameter Model:")
        print(f"  Best model: {diameter_predictor.model_name}")
        print(f"  Saved to: {args.diameter_output}")
    
    # Usage instructions
    print("\n" + "-" * 80)
    print("USAGE EXAMPLES")
    print("-" * 80)
    print("""
# Load and use tier predictor
from predictor import TierPredictor
tier_predictor = TierPredictor()
tier_predictor.load('tier_predictor_model.joblib')
tier = tier_predictor.predict("5С2Б2К1Е", "С", 12.0)

# Load and use diameter predictor
from diameter_predictor import DiameterPredictor
diameter_predictor = DiameterPredictor()
diameter_predictor.load('diameter_predictor_model.joblib')
diameter = diameter_predictor.predict("5С2Б2К1Е", "С", 12.0)
""")


if __name__ == "__main__":
    main()
