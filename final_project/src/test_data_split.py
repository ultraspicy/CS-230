import pandas as pd
import numpy as np

def split_dataset(input_file, test_rows, random_state=42):
    df = pd.read_csv(input_file)
    
    total_rows = len(df)
    if test_rows >= total_rows:
        raise ValueError(f"Requested test rows ({test_rows}) must be less than total rows ({total_rows})")
    
    np.random.seed(random_state)
    indices = np.random.permutation(total_rows)
    test_indices = indices[:test_rows]
    train_indices = indices[test_rows:]
    
    test_df = df.iloc[test_indices].reset_index(drop=True)
    train_df = df.iloc[train_indices].reset_index(drop=True)
    
    return train_df, test_df

if __name__ == "__main__":
    # File paths
    input_file = "./../resources/combined_finalized.csv"
    train_output = "./../resources/combined_finalized_train.csv"
    test_output = "./../resources/combined_finalized_test.csv"
    
    # Number of rows for test set
    test_rows = 5000
    
    try:
        # Split the dataset
        print(f"Reading from {input_file}...")
        train_df, test_df = split_dataset(input_file, test_rows)
        
        # Save the splits
        print(f"Saving training set ({len(train_df)} rows) to {train_output}")
        train_df.to_csv(train_output, index=False)
        
        print(f"Saving test set ({len(test_df)} rows) to {test_output}")
        test_df.to_csv(test_output, index=False)
        
        print("Split completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file {input_file}")
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")