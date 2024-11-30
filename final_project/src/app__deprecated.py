from transaction_predictor import TransactionPredictor
from model2 import compute_category_probabilities, print_category_probabilities
import pickle

class CombinedPredictor:
    def __init__(self, model_path: str, vocab: dict, category_to_idx: dict, customized_categories: list):
        """Initialize both predictors"""
        self.transaction_predictor = TransactionPredictor(
            model_path=model_path,
            vocab=vocab,
            category_to_idx=category_to_idx
        )
        self.customized_categories = customized_categories
    
    def predict(self, description: str, amount: float, timestamp: int) -> dict:
        """Simple predict interface"""
        # Get initial prediction
        initial_prediction = self.transaction_predictor.predict(description, amount, timestamp)
        
        # Map to custom categories
        custom_probabilities = compute_category_probabilities(
            initial_prediction['all_scores'],
            self.customized_categories
        )
        
        return {
            'original_prediction': initial_prediction,
            'custom_prediction': custom_probabilities
        }

def predict_and_map_category(
    transaction_predictor: TransactionPredictor,
    description: str,
    amount: float,
    timestamp: int,
    customized_categories: list
) -> dict:
    """
    Predict transaction category and map to custom categories
    
    Args:
        transaction_predictor: Initialized TransactionPredictor
        description: Transaction description
        amount: Transaction amount
        timestamp: Transaction timestamp
        customized_categories: List of custom categories to map to
        
    Returns:
        Dictionary containing both original and mapped predictions
    """
    # Get prediction from transaction predictor
    initial_prediction = transaction_predictor.predict(description, amount, timestamp)
    
    # Map to custom categories
    custom_probabilities = compute_category_probabilities(
        initial_prediction['all_scores'],
        customized_categories
    )
    
    return {
        'original_prediction': initial_prediction,
        'custom_prediction': custom_probabilities
    }

if __name__ == "__main__":
    # Load necessary files
    with open('./../best_model_weights/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('./../best_model_weights/categories.pkl', 'rb') as f:
        category_to_idx = pickle.load(f)
    
    # Define custom categories
    customized_categories = [
        "car",
        "dinning",
        "grocery",
        "other"
    ]
    
    print("\nExample 1: Using the combined class")
    print("-" * 50)
    
    # Initialize combined predictor
    predictor = CombinedPredictor(
        model_path='./../best_model_weights/best_model.pth',
        vocab=vocab,
        category_to_idx=category_to_idx,
        customized_categories=customized_categories
    )
    
    # Simple prediction interface
    prediction = predictor.predict(
        description="24h Fitness Annual Subs",
        amount=45.60,
        timestamp=1699123200 
    )
    print(prediction)
    
    print("\nExample 2: Using the function")
    print("-" * 50)
    
    # Initialize transaction predictor
    transaction_predictor = TransactionPredictor(
        model_path='./../best_model_weights/best_model.pth',
        vocab=vocab,
        category_to_idx=category_to_idx
    )
    
    # Make predictions using the function
    result = predict_and_map_category(
        transaction_predictor,
        description="24h Fitness Annual Subs",
        amount=45.60,
        timestamp=1699123200,
        customized_categories=customized_categories
    )
    
    # Print detailed results
    print("\nOriginal Prediction:")
    print(f"Category: {result['original_prediction']['predicted_category']}")
    print(f"Confidence: {result['original_prediction']['confidence']:.4f}")
    
    print("\nMapped Custom Categories:")
    print_category_probabilities(result['custom_prediction'])
    
    print("\nExample 3: Multiple predictions")
    print("-" * 50)
    
    # Example transactions
    transactions = [
        {
            "description": "24h Fitness Annual Subs",
            "amount": 45.60,
            "timestamp": 1699123200
        },
        {
            "description": "SHELL GAS",
            "amount": 50.25,
            "timestamp": 1699209600
        },
        {
            "description": "WALMART GROCERY",
            "amount": 125.60,
            "timestamp": 1699296000
        }
    ]
    
    for tx in transactions:
        prediction = predictor.predict(
            description=tx["description"],
            amount=tx["amount"],
            timestamp=tx["timestamp"]
        )
        print(f"\nTransaction: {tx['description']}")
        print(f"Original category: {prediction['original_prediction']['predicted_category']}")
        print("Custom categories:")
        print_category_probabilities(prediction['custom_prediction'])