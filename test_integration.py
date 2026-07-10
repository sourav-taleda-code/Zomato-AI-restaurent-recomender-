import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
import pandas as pd
import data_ingestion
import recommender

class TestIntegrationFlow(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = 'test_integration_data'
        os.makedirs(self.test_dir, exist_ok=True)
        
    def tearDown(self):
        # Clean up temporary test directories
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('data_ingestion.load_dataset')
    @patch('requests.post')
    def test_end_to_end_flow(self, mock_post, mock_load_dataset):
        # 1. Mock Hugging Face dataset response
        mock_data = {
            'train': [
                {'restaurant name': 'Rest A', 'city': 'Indiranagar', 'cuisines': 'Cafe', 'average cost for two': '500', 'rating': '4.5', 'votes': '100'},
                {'restaurant name': 'Rest B', 'city': 'Indiranagar', 'cuisines': 'Italian', 'average cost for two': '1500', 'rating': '4.8', 'votes': '200'},
                {'restaurant name': 'Rest C', 'city': 'Koramangala', 'cuisines': 'Pub', 'average cost for two': '1200', 'rating': '4.2', 'votes': '150'}
            ]
        }
        mock_load_dataset.return_value = mock_data

        # 2. Mock Groq API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "AI Summary: Rest A is the best budget match in Indiranagar."
                }
            }]
        }
        mock_post.return_value = mock_response

        # --- Phase 1: Ingest & Clean ---
        df_cleaned = data_ingestion.load_and_clean_data(output_dir=self.test_dir)
        self.assertIsNotNone(df_cleaned)
        
        target_csv = os.path.join(self.test_dir, 'cleaned_zomato_data.csv')
        self.assertTrue(os.path.exists(target_csv))

        # --- Phase 2: Load Data ---
        df_loaded = recommender.load_data(target_csv)
        self.assertEqual(len(df_loaded), 3)

        # --- Phase 3: Filter & Recommend ---
        # User query: Indiranagar, Budget: 1000
        recommendations = recommender.recommend(df_loaded, city='indiranagar', max_price=1000)
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations.iloc[0]['restaurant_name'], 'Rest A')

        # --- Phase 4: AI recommendation summary ---
        restaurants_list = recommendations.to_dict(orient='records')
        ai_summary = recommender.get_ai_recommendation(restaurants_list, 'indiranagar', 1000, api_key='fake-key')
        
        self.assertEqual(ai_summary, "AI Summary: Rest A is the best budget match in Indiranagar.")
        mock_post.assert_called_once()

if __name__ == '__main__':
    unittest.main()
