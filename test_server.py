import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
import pandas as pd
import server

class TestServerAPI(unittest.TestCase):
    
    def setUp(self):
        self.client = TestClient(server.app)

    @patch('server.df')
    def test_get_cities_endpoint(self, mock_df):
        # Setup mock DataFrame in the server module
        mock_data = {
            'city': ['btm', 'indiranagar', 'hsr']
        }
        server.df = pd.DataFrame(mock_data)
        
        response = self.client.get("/api/cities")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"cities": ["btm", "hsr", "indiranagar"]})

    @patch('server.recommend')
    @patch('server.get_ai_recommendation')
    @patch('server.df')
    def test_get_recommendations_endpoint(self, mock_df, mock_ai_rec, mock_rec):
        server.df = pd.DataFrame({'city': ['btm']})
        
        # Setup mock return values
        mock_rec.return_value = pd.DataFrame([{
            'restaurant_name': 'Test Cafe',
            'cuisines': 'Cafe',
            'average_cost_for_two': 500.0,
            'rating': 4.5,
            'votes': 100
        }])
        mock_ai_rec.return_value = "AI Insight Summary"

        response = self.client.get("/api/recommend?city=btm&max_price=500")
        
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(len(json_data["restaurants"]), 1)
        self.assertEqual(json_data["restaurants"][0]["restaurant_name"], "Test Cafe")
        self.assertEqual(json_data["ai_summary"], "AI Insight Summary")

if __name__ == '__main__':
    unittest.main()
