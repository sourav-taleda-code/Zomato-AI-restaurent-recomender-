import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import recommender

class TestRecommender(unittest.TestCase):
    
    def setUp(self):
        mock_data = {
            'restaurant_name': ['Delhi Cafe', 'Mumbai Diner', 'Delhi Fine Dine', 'Delhi Street Food'],
            'city': ['delhi', 'mumbai', 'delhi', 'delhi'],
            'average_cost_for_two': [1000.0, 1500.0, 2500.0, 300.0],
            'rating': [4.2, 4.0, 4.8, 4.5],
            'votes': [150, 80, 500, 1000]
        }
        self.mock_df = pd.DataFrame(mock_data)

    def test_filter_by_city(self):
        result = recommender.filter_restaurants(self.mock_df, city='delhi')
        self.assertEqual(len(result), 3)
        self.assertTrue(all(result['city'] == 'delhi'))

    def test_filter_by_city_case_insensitive(self):
        result = recommender.filter_restaurants(self.mock_df, city='DELHI')
        self.assertEqual(len(result), 3)

    def test_filter_by_price(self):
        result = recommender.filter_restaurants(self.mock_df, city='delhi', max_price=1200)
        self.assertEqual(len(result), 2)
        self.assertIn('Delhi Cafe', result['restaurant_name'].values)
        self.assertIn('Delhi Street Food', result['restaurant_name'].values)
        self.assertNotIn('Delhi Fine Dine', result['restaurant_name'].values)

    def test_no_results_found(self):
        self.assertTrue(recommender.filter_restaurants(self.mock_df, city='chennai').empty)
        self.assertTrue(recommender.filter_restaurants(self.mock_df, city='delhi', max_price=100).empty)

    def test_missing_api_key(self):
        res = recommender.get_ai_recommendation([{'restaurant_name': 'Test', 'cuisines': 'Indian', 'average_cost_for_two': 100, 'rating': 4.0}], 'delhi', api_key="")
        self.assertIn("Error: API key not provided.", res)

    @patch('requests.post')
    def test_successful_ai_recommendation(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Here is the AI recommendation explanation."
                }
            }]
        }
        mock_post.return_value = mock_response

        restaurants = [{
            'restaurant_name': 'BTM Biryani',
            'cuisines': 'biryani',
            'average_cost_for_two': 500.0,
            'rating': 4.5
        }]

        res = recommender.get_ai_recommendation(restaurants, 'btm', 500, api_key='fake-key')
        self.assertEqual(res, "Here is the AI recommendation explanation.")
        mock_post.assert_called_once()

if __name__ == '__main__':
    unittest.main()
