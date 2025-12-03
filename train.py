"""
Tier (Разряд) Prediction Model for Forest Elements - Training Script

This script trains a machine learning model to predict the tier (разряд) of a forest element
based on the following features:
- Состав (composition)
- Элемент леса (forest element type)
- Высота (height)

NOTE: Diameter (Диаметр) is NOT used in this model.

Usage:
    python train.py                          # Train and save model
    python train.py --data path/to/data.csv  # Train with custom data
    python train.py --output model.joblib    # Custom output path
"""

import argparse
from predictor import TierPredictor


def main():
    """Main function to train and save the tier prediction model."""
    parser = argparse.ArgumentParser(
        description='Train a tier prediction model for forest elements'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default='combined_data.csv',
        help='Path to the training data CSV file (default: combined_data.csv)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='tier_predictor_model.joblib',
        help='Path to save the trained model (default: tier_predictor_model.joblib)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("FOREST ELEMENT TIER (РАЗРЯД) PREDICTOR")
    print("Training WITHOUT diameter - using only height, composition, element")
    print("=" * 70)
    
    # Create and train predictor
    predictor = TierPredictor()
    predictor.train(args.data)
    
    # Save the model
    predictor.save(args.output)
    
    # Demonstrate example predictions
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)
    
    examples = [
        ("5С2Б2К1Е", "С", 12.0),
        ("5С2Б2К1Е", "Б", 10.0),
        ("5С2Б2К1Е", "К", 14.0),
        ("5С2Б2К1Е", "Е", 14.0),
        ("6Л2К1Е1Б+С", "Л", 19.0),
        ("7Б3Л", "Б", 2.0),
    ]
    
    print("\n{:<20} {:<10} {:<10} {:<12} {:<30}".format(
        "Состав", "Элемент", "Высота", "Predicted", "Top Probabilities"
    ))
    print("-" * 85)
    
    for composition, element, height in examples:
        prediction = predictor.predict(composition, element, height)
        probas = predictor.predict_proba(composition, element, height)
        
        # Get top 3 probabilities
        top_probas = sorted(probas.items(), key=lambda x: x[1], reverse=True)[:3]
        proba_str = ", ".join([f"T{t}:{p:.2f}" for t, p in top_probas])
        
        print(f"{composition:<20} {element:<10} {height:<10.1f} {prediction:<12} {proba_str}")
    
    print("\n" + "=" * 70)
    print(f"Training complete! Model saved to: {args.output}")
    print("=" * 70)
    
    # Show how to use the saved model
    print("\nTo use the saved model in your code:")
    print("-" * 40)
    print("""
from predictor import TierPredictor

# Load the trained model
predictor = TierPredictor()
predictor.load('tier_predictor_model.joblib')

# Make predictions
tier = predictor.predict(
    composition="5С2Б2К1Е",
    element="С",
    height=12.0
)
print(f"Predicted tier: {tier}")

# Get probabilities
probas = predictor.predict_proba(
    composition="5С2Б2К1Е",
    element="С", 
    height=12.0
)
print(f"Probabilities: {probas}")
""")


if __name__ == "__main__":
    main()

