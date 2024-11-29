# California Mall Customer Sales Dataset

import kagglehub
import os
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("captaindatasets/istanbul-mall")

print("Path to dataset files:", path)
print("\nFiles in directory:")
for root, dirs, files in os.walk(path):
    print(f"\nDirectory: {root}")
    for file in files:
        print(f"- {file}")

def load_all_data(path):
    """
    Load the mall dataset Excel files and return initial exploration
    """
    # Dictionary to store our dataframes
    dataframes = {}
    
    # Load each Excel file
    for file in os.listdir(path):
        if file.endswith('.xlsx'):
            file_path = os.path.join(path, file)
            df_name = file.replace('.xlsx', '')
            dataframes[df_name] = pd.read_excel(file_path)
            
            print(f"\n=== {df_name} ===")
            print(f"Shape: {dataframes[df_name].shape}")
            print(f"Columns: {dataframes[df_name].columns.tolist()}")
            print("\nFirst few rows:")
            print(dataframes[df_name].head())
            print("\nData Info:")
            print(dataframes[df_name].info())
            print("\n" + "="*50)
    
    return dataframes

# Download and load the data
path = kagglehub.dataset_download("captaindatasets/istanbul-mall")
dfs = load_all_data(path)


'''
Output of this file

First few rows:
  customer_id  gender   age payment_method
0     C241288  Female  28.0    Credit Card
1     C111565    Male  21.0     Debit Card
2     C266599    Male  20.0           Cash
3     C988172  Female  66.0    Credit Card
4     C189076  Female  53.0           Cash

Data Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 99457 entries, 0 to 99456
Data columns (total 4 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   customer_id     99457 non-null  object 
 1   gender          99457 non-null  object 
 2   age             99338 non-null  float64
 3   payment_method  99457 non-null  object 
dtypes: float64(1), object(3)
memory usage: 3.0+ MB
None
'''