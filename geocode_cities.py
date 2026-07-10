import pandas as pd
import requests
import json
import time
import os

df = pd.read_csv('data/cleaned_zomato_data.csv')
cities = sorted(df['city'].dropna().unique().tolist())

coords = {}
for city in cities:
    # Adding Bangalore to ensure we get locations in Bangalore
    query = f"{city}, Bangalore, India"
    url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(query)}&format=json&limit=1"
    headers = {'User-Agent': 'ZomatoAI-App/1.0'}
    try:
        res = requests.get(url, headers=headers)
        data = res.json()
        if data:
            coords[city] = {'lat': float(data[0]['lat']), 'lon': float(data[0]['lon'])}
            print(f"Found {city}: {coords[city]}")
        else:
            print(f"Not found: {city}")
    except Exception as e:
        print(f"Error {city}: {e}")
    time.sleep(1) # Respect Nominatim rate limits

with open('data/city_coords.json', 'w') as f:
    json.dump(coords, f)
print("Done.")
