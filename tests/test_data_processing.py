import sys
import os
import unittest
from unittest.mock import patch
import pandas as pd

# Add the src directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_processing import load_and_clean_data, preprocess_data

class TestDataProcessing(unittest.TestCase):
    
    # Define the path to the test data file
    test_data_path = r'C:\Users\karan\OneDrive\Desktop\KARAN_SINGH_Workplete_ai_Assignment\workplete_ai_employee\data\tests_data.csv'
    
    @patch('data_processing.pd.read_csv')
    def test_load_and_clean_data(self, mock_read_csv):
        # Mocking the return value of the pandas read_csv function
        mock_read_csv.return_value = pd.DataFrame({'Total': [1, 2, 3]})
        
        # Calling the function to test
        df = load_and_clean_data(self.test_data_path)
        
        # Check if the return value is a DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        # Check if the DataFrame contains the expected column
        self.assertIn('Total', df.columns)
    
    def test_preprocess_data(self):
        # Test data for preprocessing
        df = pd.DataFrame({'Total': [1, 2, 3]})
        target_column = 'Total'
        
        # Calling the function to test
        df_processed = preprocess_data(df, target_column)
        
        # Check if the DataFrame contains the expected column
        self.assertIn('Total', df_processed.columns)

if __name__ == '__main__':
    unittest.main()
