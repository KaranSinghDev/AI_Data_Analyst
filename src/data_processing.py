import pandas as pd
import json



import pandas as pd
import json

def load_and_clean_data(file_path):
    """
    Load the dataset from a file and clean it.
    Supports CSV, JSON, and Excel file formats.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            # Determine the structure of the JSON data
            if isinstance(data, dict):
                if 'records' in data:
                    df = pd.json_normalize(data['records'], sep='_')
                else:
                    df = pd.json_normalize(data, sep='_')
            elif isinstance(data, list):
                df = pd.json_normalize(data, sep='_')
            else:
                print("Unsupported JSON format. Please provide a list of records or a dictionary with records.")
                return None
        
        elif file_path.endswith('.xls') or file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            print("Unsupported file format. Please use CSV, JSON, or Excel.")
            return None

        # Basic cleaning
        df.dropna(inplace=True)  # Remove rows with missing values
        df.columns = df.columns.str.strip()  # Clean up any extra spaces in column names

        print("Data successfully loaded and cleaned.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def preprocess_data(df, target_column):
    """
    Preprocess the dataset: Clean and prepare data.
    If the target column contains dictionaries, try to flatten them.
    """
    try:
        # Check if the target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
        
        # Check if the values in the target column are dictionaries
        if isinstance(df[target_column].iloc[0], dict):
            print(f"Target column '{target_column}' contains nested dictionaries. Flattening data...")

            # Flatten the nested dictionaries by expanding them into separate columns
            # Each dictionary will be expanded into multiple columns
            dict_df = pd.json_normalize(df[target_column])
            
            # Combine the original DataFrame (excluding the target column) with the flattened dictionary DataFrame
            df = df.drop(columns=[target_column])
            df = pd.concat([df, dict_df], axis=1)

            print(f"Data after flattening: {df.head()}")

        # Return the processed DataFrame
        return df
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None
