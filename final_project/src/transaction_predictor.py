import torch
import numpy as np
from model import TransactionClassifier

class TransactionPredictor:
    def __init__(self, model_path: str, vocab: dict, category_to_idx: dict):
        """
        Initialize predictor with trained model and necessary mappings
        
        Args:
            model_path: Path to the saved model weights
            vocab: Dictionary mapping words to indices
            category_to_idx: Dictionary mapping categories to indices
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab = vocab
        self.category_to_idx = category_to_idx
        self.idx_to_category = {idx: cat for cat, idx in category_to_idx.items()}
        
        # Initialize and load the model
        input_dim = len(vocab) + 3  # vocab size + numerical features
        num_categories = len(category_to_idx)
        self.model = TransactionClassifier(input_dim, num_categories)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    def _vectorize_description(self, description: str) -> torch.Tensor:
        """Convert a single description to term frequency vector"""
        vector = torch.zeros(len(self.vocab))
        for word in description.lower().split():
            if word in self.vocab:
                vector[self.vocab[word]] += 1
        return vector
    
    def predict(self, description: str, amount: float, timestamp: int) -> dict:
        """
        Make prediction for a single transaction
        
        Args:
            description: Transaction description text
            amount: Transaction amount
            timestamp: Unix timestamp of the transaction
            
        Returns:
            Dictionary containing predicted category and confidence scores
        """
        # Prepare features
        desc_vector = self._vectorize_description(description)
        day_of_week = timestamp % 7
        day_of_month = timestamp % 30
        
        # Combine features
        features = torch.cat([
            desc_vector,
            torch.tensor([amount, day_of_week, day_of_month], dtype=torch.float32)
        ])
        
        # Make prediction
        with torch.no_grad():
            features = features.to(self.device).unsqueeze(0)  # Add batch dimension
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_idx = probabilities.argmax().item()
            
        # Get confidence scores for all categories
        scores = {
            self.idx_to_category[idx]: score.item() 
            for idx, score in enumerate(probabilities)
        }
        
        return {
            'predicted_category': self.idx_to_category[predicted_idx],
            'confidence': probabilities[predicted_idx].item(),
            'all_scores': scores
        }
    
    def predict_batch(self, descriptions: list, amounts: list, timestamps: list) -> list:
        """
        Make predictions for multiple transactions
        
        Args:
            descriptions: List of transaction descriptions
            amounts: List of transaction amounts
            timestamps: List of transaction timestamps
            
        Returns:
            List of prediction dictionaries
        """
        # Prepare batch features
        batch_size = len(descriptions)
        features = torch.zeros(batch_size, len(self.vocab) + 3)
        
        for i in range(batch_size):
            desc_vector = self._vectorize_description(descriptions[i])
            day_of_week = timestamps[i] % 7
            day_of_month = timestamps[i] % 30
            
            features[i] = torch.cat([
                desc_vector,
                torch.tensor([amounts[i], day_of_week, day_of_month], 
                           dtype=torch.float32)
            ])
        
        # Make predictions
        with torch.no_grad():
            features = features.to(self.device)
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_indices = probabilities.argmax(dim=1)
        
        # Prepare results
        results = []
        for i in range(batch_size):
            scores = {
                self.idx_to_category[idx]: score.item() 
                for idx, score in enumerate(probabilities[i])
            }
            results.append({
                'predicted_category': self.idx_to_category[predicted_indices[i].item()],
                'confidence': probabilities[i][predicted_indices[i]].item(),
                'all_scores': scores
            })
        
        return results

# Example usage
def example_usage():
    # Assume we have a trained model and saved mappings
    predictor = TransactionPredictor(
        model_path='best_model.pth',
        vocab=dataset.vocab,  # from your training dataset
        category_to_idx=dataset.category_to_idx  # from your training dataset
    )
    
    # Single prediction
    prediction = predictor.predict(
        description="WALMART GROCERY",
        amount=125.60,
        timestamp=1699123200
    )
    print("Single prediction:", prediction)
    
    # Batch prediction
    batch_predictions = predictor.predict_batch(
        descriptions=["WALMART GROCERY", "SHELL GAS", "STARBUCKS COFFEE"],
        amounts=[125.60, 45.30, 5.75],
        timestamps=[1699123200, 1699296000, 1699382400]
    )
    print("Batch predictions:", batch_predictions)

if __name__ == "__main__":
    import pickle
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('categories.pkl', 'rb') as f:
        category_to_idx = pickle.load(f)

    # Initialize predictor
    predictor = TransactionPredictor(
        model_path='best_model.pth',
        vocab=vocab,
        category_to_idx=category_to_idx
    )

    # Make predictions
    prediction = predictor.predict(
        description="24h Fitness Annual Subs",
        amount=45.60,
        timestamp=1699123200  # Unix timestamp
    )
    print(prediction)