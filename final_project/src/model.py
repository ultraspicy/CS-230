import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict
import csv
from collections import Counter
from datetime import datetime

class TransactionDataset(Dataset):
    """load dataset for financial transactions"""
    def __init__(self, descriptions: List[str], amounts: List[float], 
                 timestamps: List[int], categories: List[str]):
        self.descriptions = descriptions
        self.amounts = torch.tensor(amounts, dtype=torch.float32)
        
        # Extract temporal features from timestamps
        self.day_of_week, self.day_of_month = self.process_dates(timestamps)
        
        # Convert categories to indices, e.g.
        #    {
        #        'food': 0,
        #        'electronics': 1,
        #        'books': 2
        #    }
        unique_categories = sorted(set(categories))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
        # print(category_to_idx)  # {'Gas': 0, 'Groceries': 1, 'Restaurant': 2}
        self.labels = torch.tensor([self.category_to_idx[cat] for cat in categories])
        # print(labels)  # tensor([1, 0, 2])
        
        # Convert descriptions to term frequency vectors
        self.vocab = self._build_vocabulary(descriptions)
        self.description_vectors = self._vectorize_descriptions(descriptions)

    def process_dates(self, dates):
        """
        Convert dates to day_of_week and day_of_month features
        Handles both timestamp (int/float) and date strings (e.g., '2024-11-08')
        
        Args:
            dates: List of dates in either timestamp or string format
            
        Returns:
            tuple: (day_of_week tensor, day_of_month tensor)
        """
        processed_timestamps = []
        
        for date in dates:
            if isinstance(date, (int, float)):  # If already timestamp
                timestamp = date
            else:  # If string date
                try:
                    # Try parsing as ISO format (YYYY-MM-DD)
                    dt = datetime.strptime(date, '%Y-%m-%d')
                    timestamp = dt.timestamp()
                except ValueError as e:
                    print(f"Warning: Could not parse date {date}: {e}")
                    # Use a default timestamp if parsing fails
                    timestamp = 0
            
            processed_timestamps.append(timestamp)
        
        timestamps = np.array(processed_timestamps)
        
        # Convert to datetime objects for accurate day extraction
        dt_objects = [datetime.fromtimestamp(ts) for ts in timestamps]
        
        # Extract day of week (0 = Monday, 6 = Sunday)
        day_of_week = np.array([dt.weekday() for dt in dt_objects])
        
        # Extract day of month (1-31)
        day_of_month = np.array([dt.day for dt in dt_objects])
        
        return (torch.tensor(day_of_week, dtype=torch.float32),
                torch.tensor(day_of_month, dtype=torch.float32))
   
    def _build_vocabulary(self, descriptions: List[str], max_features: int = 1000):
        """Build a simple vocabulary from descriptions"""
        word_freq = {}
        for desc in descriptions:
            for word in desc.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1
                
        # Keep most frequent words
        vocab = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_features]
        return {word: idx for idx, (word, _) in enumerate(vocab)}
    
    def _vectorize_descriptions(self, descriptions: List[str]):
        """Convert descriptions to term frequency vectors"""
        vectors = torch.zeros((len(descriptions), len(self.vocab)))
        for i, desc in enumerate(descriptions):
            for word in desc.lower().split():
                if word in self.vocab:
                    vectors[i, self.vocab[word]] += 1
        return vectors
    
    def __len__(self):
        return len(self.descriptions)
    
    def __getitem__(self, idx):
        features = torch.cat([
            self.description_vectors[idx],
            torch.tensor([
                self.amounts[idx],
                self.day_of_week[idx],
                self.day_of_month[idx]
            ])
        ])
        return features, self.labels[idx]

class TransactionClassifier(nn.Module):
    def __init__(self, input_dim: int, num_categories: int):
        super().__init__()
        
        # Architecture for edge deployment
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            
            nn.Linear(32, num_categories)
        )
        
    def forward(self, x):
        return self.layers(x)

def train_model(model: nn.Module, train_loader: DataLoader, 
                val_loader: DataLoader, num_epochs: int = 100, 
                learning_rate: float = 0.01):
    """Training loop with validation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase 
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Disables gradient calculation for efficiency
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate accuracy and average loss
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # # Early stopping
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     patience_counter = 0
        #     # Save best model here if needed
        #     torch.save(model.state_dict(), 'best_model.pth')
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print(f'Early stopping triggered after epoch {epoch+1}')
        #         break
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'Validation Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%')
        
    return model

# Example usage
def prepare_example_data(mode = 'fake'):
    if mode == 'fake':
        return prepare_example_data_fake()
    elif mode == 'mini':
        return prepare_example_data_mini()
    

def prepare_example_data_fake():
    """Prepare sample data for demonstration"""
    descriptions = [
        "WALMART GROCERY",
        "AMAZON PRIME",
        "SHELL GAS",
        "STARBUCKS COFFEE",
        "RENT PAYMENT"
    ]
    amounts = [125.60, 14.99, 45.30, 5.75, 1200.00]
    timestamps = [1699123200, 1699209600, 1699296000, 1699382400, 1699468800]
    categories = ["Groceries", "Subscription", "Transportation", "Food", "Housing"]
    return descriptions, amounts, timestamps, categories

def prepare_example_data_mini():
    """ read csv file exported from copiolt for fast verification"""
    column_data = process_csv_with_analysis('./../resources/combined_finalized.csv')
    analyze_column_data(column_data)

    # Convert amount from string to float
    amounts = [float(amt) for amt in column_data['amount']]
    return column_data['description'], amounts,  column_data['date'], column_data['category']


def process_csv_with_analysis(file_path: str) -> Dict[str, List[str]]:
    """
    Read CSV file and create a mapping of column names to their values with additional analysis
    """
    column_map: Dict[str, List[str]] = {}
    
    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        
        # Initialize lists for each column
        for column in csv_reader.fieldnames or []:
            column_map[column] = []
            
        # Read each row
        for row in csv_reader:
            for column, value in row.items():
                column_map[column].append(value)
    
    return column_map

def analyze_column_data(column_map: Dict[str, List[str]]):
    """Print detailed analysis of each column"""
    for column, values in column_map.items():
        print(f"\n=== {column} ===")
        print(f"Total entries: {len(values)}")
        
        # Count non-empty values
        non_empty = [v for v in values if v.strip()]
        print(f"Non-empty entries: {len(non_empty)}")
        
        # Show unique values and their counts
        value_counts = Counter(values)
        print(f"Unique values: {len(value_counts)}")
        
        # Show most common values
        if len(value_counts) > 1:
            print("\nMost common values:")
            for value, count in value_counts.most_common(3):
                print(f"  {value!r}: {count} times")

def print_model_info(model: nn.Module):
    """Print detailed information about model architecture and parameters"""
    print("\n=== Model Architecture and Parameters ===")
    
    # Print overall architecture
    print("\nModel Architecture:")
    print(model)
    
    # Calculate and print total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Print detailed layer information
    print("\nDetailed Layer Information:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"\nLayer: {name}")
            print(f"Shape: {param.shape}")
            print(f"Parameters: {param.numel():,}")
            
            # Print parameter statistics
            if param.numel() > 0:
                print(f"Mean: {param.mean().item():.6f}")
                print(f"Std: {param.std().item():.6f}")
                print(f"Min: {param.min().item():.6f}")
                print(f"Max: {param.max().item():.6f}")

if __name__ == "__main__":
    # Add flag for training
    TRAIN_MODEL = False  # Set to False to skip training
    
    # Prepare example data
    descriptions, amounts, timestamps, categories = prepare_example_data('mini')
    
    # Create dataset
    dataset = TransactionDataset(descriptions, amounts, timestamps, categories)
    
    # Initialize model
    input_dim = len(dataset.vocab) + 3  # vocab size + numerical features
    num_categories = len(dataset.category_to_idx)
    model = TransactionClassifier(input_dim, num_categories)
    
    # Print initial model information
    print("\n=== Initial Model Parameters ===")
    print_model_info(model)
    
    if TRAIN_MODEL:
        # Split into train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        # Train model
        trained_model = train_model(model, train_loader, val_loader)
        
        # Print model information after training
        print("\n=== Model Parameters After Training ===")
        print_model_info(trained_model)

        # Save model and mappings
        torch.save(model.state_dict(), 'best_model.pth')
        import pickle
        with open('vocab.pkl', 'wb') as f:
            pickle.dump(dataset.vocab, f)
        with open('categories.pkl', 'wb') as f:
            pickle.dump(dataset.category_to_idx, f)
        print('==========================================================')
        print('==============  training complete ========================')
    else:
        print("\nSkipping training phase. Only showing model architecture and initial parameters.")