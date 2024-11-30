from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

def compute_category_probabilities(
    category_scores: Dict[str, float],
    customized_categories: List[str],
    model_name: str = 'all-MiniLM-L6-v2'
) -> Dict[str, float]:
    """
    Computes probabilities for customized categories based on original scores and semantic similarity.
    
    Args:
        category_scores: Dictionary of original categories and their probabilities
        customized_categories: List of target categories to map to
        model_name: Name of the sentence transformer model to use
    
    Returns:
        Dictionary mapping customized categories to their computed probabilities
    """
    # Initialize the model
    model = SentenceTransformer(model_name)
    
    # Remove the empty string key if it exists
    filtered_scores = category_scores.copy()
    filtered_scores.pop('', None)  # Safely remove '' key if it exists
    
    # Prepare original categories and scores
    original_categories = list(filtered_scores.keys())
    original_probabilities = np.array([filtered_scores[cat] for cat in original_categories])
    
    # Renormalize probabilities
    total_prob = np.sum(original_probabilities)
    if total_prob > 0:
        original_probabilities = original_probabilities / total_prob
    
    # Get embeddings
    original_embeddings = model.encode(original_categories)
    custom_embeddings = model.encode(customized_categories)
    
    # Convert to torch tensors
    orig_emb = torch.tensor(original_embeddings)
    cust_emb = torch.tensor(custom_embeddings)
    
    # Calculate cosine similarity matrix
    similarity = F.cosine_similarity(
        orig_emb.unsqueeze(1),
        cust_emb.unsqueeze(0),
        dim=2
    )
    
    # Convert similarity to numpy for easier manipulation
    similarity_matrix = similarity.numpy()
    
    # Compute weighted probabilities for each custom category
    custom_probabilities = {}
    for i, custom_cat in enumerate(customized_categories):
        # Multiply similarities by original probabilities and sum
        weighted_prob = np.sum(similarity_matrix[:, i] * original_probabilities)
        custom_probabilities[custom_cat] = float(weighted_prob)
    
    # Normalize probabilities to sum to 1
    total_prob = sum(custom_probabilities.values())
    if total_prob > 0:
        custom_probabilities = {
            k: v/total_prob for k, v in custom_probabilities.items()
        }
    
    return custom_probabilities

def print_category_probabilities(scores: Dict[str, float]) -> None:
    """Pretty print the category probabilities."""
    print("\nCategory Probabilities:")
    print("-" * 50)
    print(f"{'Category':<15} {'Probability':<10}")
    print("-" * 50)
    
    # Sort by probability in descending order
    sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for cat, prob in sorted_cats:
        print(f"{cat:<15} {prob:.6f}")

# Example usage
test_scores = {
    '': 0.9999305009841919,  # This will be excluded
    'Car & Gas': 8.64992349665708e-14,
    'Coffee Shop': 5.76815513976108e-16,
    'Date': 1.7026805583953558e-13,
    'Entertainment & Streaming': 7.021092495737848e-18,
    'Fast Food': 3.844865404254523e-25,
    'Financial Services': 6.639378310373603e-13,
    'Fitness & Wellness': 5.248391710272659e-19,
    'Groceries': 5.104009840708784e-1,
    'Healthcare': 1.7476949087525574e-18,
    'Hobbies & Recreation': 3.465092143813543e-16,
    'Housing': 3.8260450827118955e-20,
    'Insurance': 5.447913170933405e-13,
    'Personal Care & Beauty': 3.5481407795288465e-18,
    'Professional Services': 4.2377468322523743e-19,
    'Rent': 6.811835191911086e-05,
    'Restaurants': 1.716164189924374e-11,
    'Subscriptions': 1.8517478539035667e-15,
    'Technology & Electronics': 6.684154948875092e-18,
    'Transportation': 1.0830636387737513e-10,
    'Travel & Vacations': 2.0414567378391487e-16,
    'Utilities': 1.5073000424301215e-14,
    'Utilities & Communications': 9.868993643136204e-14,
    'miscellaneous': 1.3628445003632805e-06
}

customized_categories = [
    "car",
    "dinning",
    "grocery",
    "other"
]

# Compute and print results
custom_probabilities = compute_category_probabilities(test_scores, customized_categories)
print_category_probabilities(custom_probabilities)