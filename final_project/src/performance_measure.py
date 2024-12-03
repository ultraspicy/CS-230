import pandas as pd
import time
from datetime import datetime
import pickle
from transaction_predictor import get_memory_usage, TransactionPredictor

import pandas as pd
import time
from datetime import datetime
import pickle
from collections import defaultdict

def compute_metrics_per_category(predictions):
    """
    Compute precision, recall, and F1-score for each category
    
    Returns:
        Dictionary with metrics for each category
    """
    # Initialize counters
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    
    # Count all categories
    all_categories = set()
    for p in predictions:
        all_categories.add(p['actual_category'])
        all_categories.add(p['predicted_category'])
    
    # Calculate TP, FP, FN for each category
    for pred in predictions:
        actual = pred['actual_category']
        predicted = pred['predicted_category']
        
        if actual == predicted:
            true_positives[actual] += 1
        else:
            false_positives[predicted] += 1
            false_negatives[actual] += 1
    
    # Calculate metrics for each category
    metrics = {}
    for category in all_categories:
        tp = true_positives[category]
        fp = false_positives[category]
        fn = false_negatives[category]
        
        # Calculate precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Calculate recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[category] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn  # total number of actual instances
        }
    
    return metrics

def print_metrics_table(metrics):
    """Print metrics in a formatted table"""
    print("\n=== Per-Category Performance Metrics ===")
    print(f"{'Category':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 72)
    
    # Calculate macro average
    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    total_support = 0
    
    # Print metrics for each category (without sorting)
    for category, m in metrics.items():
        print(f"{category:<30} {m['precision']:>9.2%} {m['recall']:>9.2%} "
              f"{m['f1']:>9.2%} {m['support']:>10d}")
        
        macro_precision += m['precision']
        macro_recall += m['recall']
        macro_f1 += m['f1']
        total_support += m['support']
    
    # Calculate and print macro averages
    num_categories = len(metrics)
    print("-" * 72)
    print(f"{'Macro Average':<30} {macro_precision/num_categories:>9.2%} "
          f"{macro_recall/num_categories:>9.2%} {macro_f1/num_categories:>9.2%} "
          f"{total_support:>10d}")

def predict_from_csv(csv_path, predictor):
    """
    Read transactions from CSV and make predictions using the TransactionPredictor
    
    Args:
        csv_path: Path to the CSV file
        predictor: Initialized TransactionPredictor instance
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert dates to timestamps
    df['timestamp'] = pd.to_datetime(df['date']).apply(lambda x: int(time.mktime(x.timetuple())))
    
    # Process each transaction
    predictions = []
    for idx, row in df.iterrows():
        prediction = predictor.predict_predefined(
            description=row['description'],
            amount=float(row['amount']),
            timestamp=row['timestamp']
        )
        
        # Combine prediction with original transaction data
        result = {
            'date': row['date'],
            'description': row['description'],
            'amount': row['amount'],
            'predicted_category': prediction['predicted_category'],
            'confidence': prediction['confidence'],
            'actual_category': row['category'],
            'all_scores': prediction['all_scores']
        }
        predictions.append(result)
        
        # Print progress and prediction
        print(f"\nTransaction {idx + 1}:")
        print(f"Description: {row['description']}")
        print(f"Amount: ${row['amount']}")
        print(f"Actual Category: {row['category']}")
        print(f"Predicted Category: {prediction['predicted_category']}")
        print(f"Confidence: {prediction['confidence']:.2%}")
        
    return predictions

def print_misclassified_transactions(predictions):
    """Print detailed analysis of misclassified transactions"""
    print("\n=== Misclassified Transactions Analysis ===")
    print("Format: Description (Amount) -> Predicted [Actual] (Confidence)")
    print("-" * 80)
    
    misclassified = [p for p in predictions if p['predicted_category'] != p['actual_category']]
    
    for idx, pred in enumerate(misclassified, 1):
        print(f"{idx}. {pred['description']} (${pred['amount']}) -> "
              f"{pred['predicted_category']} [{pred['actual_category']}] "
              f"({pred['confidence']:.2%})")
        # Print top 3 predictions with their scores
        sorted_scores = sorted(pred['all_scores'].items(), key=lambda x: x[1], reverse=True)[:3]
        print("   Top 3 predictions:")
        for category, score in sorted_scores:
            print(f"   - {category}: {score:.2%}")
        print("-" * 80)
    
    return len(misclassified)

if __name__ == "__main__":
    # Load vocabulary and category mappings
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    with open('./../best_model_weights/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('./../best_model_weights/categories.pkl', 'rb') as f:
        category_to_idx = pickle.load(f)
    
    vocab_loaded_memory = get_memory_usage()
    print(f"Memory after loading vocab/categories: {vocab_loaded_memory - initial_memory:.2f} MB")
    
    # Initialize predictor
    predictor = TransactionPredictor(
        model_path='./../best_model_weights/best_model.pth',
        vocab=vocab,
        category_to_idx=category_to_idx
    )
    
    # Process transactions and make predictions
    predictions = predict_from_csv('./../resources/combined_finalized_test.csv', predictor)
    
    # Print misclassified transactions
    num_misclassified = print_misclassified_transactions(predictions)
    
    # Calculate and print metrics per category
    metrics = compute_metrics_per_category(predictions)
    print_metrics_table(metrics)
    
    # Calculate and print overall statistics
    total = len(predictions)
    correct = total - num_misclassified
    accuracy = correct / total
    
    print(f"\nOverall Statistics:")
    print(f"Total Transactions: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Misclassified: {num_misclassified}")
    print(f"Accuracy: {accuracy:.2%}")
    
    final_memory = get_memory_usage()
    print(f"\nFinal memory usage: {final_memory:.2f} MB")
    print(f"Total memory increase: {final_memory - initial_memory:.2f} MB")