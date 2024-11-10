import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict

class TransactionDataset(Dataset):
    """Simple dataset for financial transactions"""
    def __init__(self, descriptions: List[str], amounts: List[float], 
                 timestamps: List[int], categories: List[str]):
        self.descriptions = descriptions
        self.amounts = torch.tensor(amounts, dtype=torch.float32)
        
        # Extract temporal features from timestamps
        timestamps = np.array(timestamps)
        self.day_of_week = torch.tensor(timestamps % 7, dtype=torch.float32)
        self.day_of_month = torch.tensor(timestamps % 30, dtype=torch.float32)
        
        # Convert categories to indices
        unique_categories = sorted(set(categories))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
        self.labels = torch.tensor([self.category_to_idx[cat] for cat in categories])
        
        # Convert descriptions to term frequency vectors
        self.vocab = self._build_vocabulary(descriptions)
        self.description_vectors = self._vectorize_descriptions(descriptions)
        
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
            
            nn.Linear(64, num_categories)
        )
        
    def forward(self, x):
        return self.layers(x)

def train_model(model: nn.Module, train_loader: DataLoader, 
                val_loader: DataLoader, num_epochs: int = 10, 
                learning_rate: float = 0.001):
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
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model here if needed
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'Validation Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%')
        
    return model

# Example usage
def prepare_example_data():
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

if __name__ == "__main__":
    # Prepare example data
    descriptions, amounts, timestamps, categories = prepare_example_data()
    
    # Create dataset
    dataset = TransactionDataset(descriptions, amounts, timestamps, categories)
    
    # Split into train/val (in practice, use proper splitting)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model
    input_dim = len(dataset.vocab) + 3  # vocab size + numerical features
    num_categories = len(dataset.category_to_idx)
    model = TransactionClassifier(input_dim, num_categories)
    
    # Train model
    trained_model = train_model(model, train_loader, val_loader)

    # After training
    torch.save(model.state_dict(), 'best_model.pth')
    # Save vocab and category mappings
    import pickle
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(dataset.vocab, f)
    with open('categories.pkl', 'wb') as f:
        pickle.dump(dataset.category_to_idx, f)