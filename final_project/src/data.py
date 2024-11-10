# California Mall Customer Sales Dataset

import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("captaindatasets/istanbul-mall")

print("Path to dataset files:", path)

def load_mall_data(path):
    """
    Load the mall dataset from the given path and return initial exploration
    """
    # Find the CSV file in the downloaded directory
    csv_file = None
    for file in os.listdir(path):
        if file.endswith('.csv'):
            csv_file = os.path.join(path, file)
            break
    
    if csv_file is None:
        raise FileNotFoundError("No CSV file found in the specified path")
    
    # Read the dataset
    df = pd.read_csv(csv_file)
    
    # Display basic information
    print("\nDataset Shape:", df.shape)
    print("\nColumns in the dataset:", df.columns.tolist())
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
    print("\nBasic information about the dataset:")
    print(df.info())
    
    return df

# Load and explore the data
df = load_mall_data(path)