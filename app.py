import os
from recommender import load_data, recommend, get_ai_recommendation
from data_ingestion import load_and_clean_data

def main():
    csv_path = 'data/cleaned_zomato_data.csv'
    
    # Step 1 Integration: Auto-ingest if missing
    if not os.path.exists(csv_path):
        print("Cleaned dataset not found. Ingesting dataset from Hugging Face...")
        df = load_and_clean_data()
        if df is None:
            print("Failed to ingest data.")
            return
    else:
        df = load_data(csv_path)

    # Display areas
    cities = sorted(df['city'].dropna().unique())
    print("Available Cities/Areas:")
    print(", ".join(cities))
    print("-" * 50)

    # Step 2: User Input
    city = input("Enter City: ").strip()
    price_input = input("Enter Max Budget (press Enter to skip): ").strip()
    max_price = float(price_input) if price_input else None

    # Step 3 & 4: Integrate and Recommend
    results = recommend(df, city, max_price)
    
    # Step 5: Display to the user
    if results.empty:
        print("\nNo recommendations found matching your criteria.")
    else:
        print("\n--- Recommendations ---")
        show_cols = [c for c in ['restaurant_name', 'cuisines', 'average_cost_for_two', 'rating'] if c in results.columns]
        print(results[show_cols].to_string(index=False))
        
        print("\nFetching AI Recommendations from Groq...")
        restaurants_list = results.to_dict(orient='records')
        ai_summary = get_ai_recommendation(restaurants_list, city, max_price)
        print("\n--- AI Summary ---")
        print(ai_summary)

if __name__ == '__main__':
    main()
