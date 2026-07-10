from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
from recommender import load_data, recommend, get_ai_recommendation

app = FastAPI(title="Zomato AI Recommendation API")

# Enable CORS so our frontend can call the backend API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

csv_path = 'data/cleaned_zomato_data.csv'
df = load_data(csv_path)

@app.get("/api/cities")
def get_cities():
    if df.empty:
        return {"cities": []}
    cities = sorted(df['city'].dropna().unique().tolist())
    return {"cities": cities}

@app.get("/api/recommend")
def get_recommendations(
    city: str = Query(..., description="The city/area to search in"),
    max_price: float = Query(None, description="Max cost for two people"),
    diet: str = Query(None, description="Diet preference (veg/non-veg)")
):
    if df.empty:
        return {"error": "Dataset missing. Please run data_ingestion.py."}

    results = recommend(df, city, max_price, diet=diet)
    
    if results.empty:
        return {"restaurants": [], "ai_summary": "No matching restaurants found."}
        
    restaurants_list = results.to_dict(orient='records')
    ai_summary = get_ai_recommendation(restaurants_list, city, max_price)
    
    return {
        "restaurants": restaurants_list,
        "ai_summary": ai_summary
    }

# Serve the static HTML frontend
@app.get("/")
def read_root():
    return FileResponse("index.html")
