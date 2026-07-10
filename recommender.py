import pandas as pd
import os

def load_data(data_path='data/cleaned_zomato_data.csv'):
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        return pd.DataFrame()

def filter_restaurants(df, city, max_price=None):
    if df.empty:
        return df
    filtered_df = df[df['city'] == city.lower().strip()]
    if max_price is not None and 'average_cost_for_two' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['average_cost_for_two'] <= float(max_price)]
    return filtered_df

def recommend(df, city, max_price=None, top_n=5):
    filtered_df = filter_restaurants(df, city, max_price)
    if filtered_df.empty:
        return pd.DataFrame()
    sort_cols = [col for col in ['rating', 'votes'] if col in filtered_df.columns]
    if sort_cols:
        filtered_df = filtered_df.fillna({col: 0 for col in sort_cols})
        filtered_df = filtered_df.sort_values(by=sort_cols, ascending=False)
    return filtered_df.head(top_n)

def get_ai_recommendation(restaurants, query_city, query_price=None, api_key=None):
    import requests
    
    if api_key is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                with open(os.path.join(base_dir, "secrets.txt"), "r") as f:
                    api_key = f.read().strip()
            except FileNotFoundError:
                pass
    if not api_key:
        return "Error: API key not provided."
        
    if not restaurants:
        return "No restaurants available to recommend."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    restaurants_text = ""
    for idx, r in enumerate(restaurants[:5]):
        restaurants_text += f"{idx+1}. {r['restaurant_name']} - Cuisine: {r['cuisines']}, Avg Cost for Two: {r['average_cost_for_two']}, Rating: {r['rating']}\n"
         
    prompt = (
        f"A user is looking for restaurant recommendations in '{query_city}' with a budget of '{query_price}'.\n"
        f"Here are the top matches:\n{restaurants_text}\n"
        f"Summarize why these choices are recommended and highlight the best one."
    )
    
    # ponytail: use active llama-3.1-8b-instant model to fix 400 bad request error
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a helpful Zomato AI assistant recommending restaurants."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Failed to get AI recommendation: {e}"
