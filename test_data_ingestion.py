import unittest
from unittest.mock import patch
import pandas as pd
import data_ingestion

class TestDataIngestion(unittest.TestCase):

    @patch('data_ingestion.load_dataset')
    @patch('data_ingestion.os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_load_and_clean_data(self, mock_to_csv, mock_makedirs, mock_load_dataset):
        mock_data = {
            'train': [
                {
                    'restaurant name': 'Test Rest 1', 
                    'city': ' Delhi ', 
                    'cuisines': 'Indian, Chinese', 
                    'average cost for two': '₹ 1,500', 
                    'rating': '4.5'
                },
                {
                    'restaurant name': 'Missing City Rest', 
                    'city': None, 
                    'cuisines': 'Cafe', 
                    'average cost for two': '500', 
                    'rating': 'NEW'
                },
                {
                    'restaurant name': 'Test Rest 2', 
                    'city': 'mumbai', 
                    'cuisines': None, 
                    'average cost for two': '2,000', 
                    'rating': '-'
                }
            ]
        }
        mock_load_dataset.return_value = mock_data
        
        df = data_ingestion.load_and_clean_data(output_dir='dummy_data')
        
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertIn('restaurant_name', df.columns)
        self.assertIn('average_cost_for_two', df.columns)
        self.assertEqual(df.iloc[0]['city'], 'delhi')
        self.assertEqual(df.iloc[1]['city'], 'mumbai')
        self.assertEqual(df.iloc[0]['cuisines'], 'indian, chinese')
        self.assertEqual(df.iloc[1]['cuisines'], 'unknown')
        self.assertEqual(df.iloc[0]['average_cost_for_two'], 1500.0)
        self.assertEqual(df.iloc[1]['average_cost_for_two'], 2000.0)
        self.assertEqual(df.iloc[0]['rating'], 4.5)
        self.assertTrue(pd.isna(df.iloc[1]['rating']))

if __name__ == '__main__':
    unittest.main()
