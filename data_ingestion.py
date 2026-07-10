import pandas as pd
from datasets import load_dataset
import os

def load_and_clean_data(output_dir='data'):
    print("Loading dataset from Hugging Face (ManikaSaini/zomato-restaurant-recommendation)...")
    try:
        dataset = load_dataset("ManikaSaini/zomato-restaurant-recommendation")
        df = pd.DataFrame(dataset['train'])
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # --- Data Cleaning & Preprocessing ---
    df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]
    
    rename_map = {
        'name': 'restaurant_name',
        'rate': 'rating',
        'approx_cost(for_two_people)': 'average_cost_for_two',
        'listed_in(city)': 'city'
    }
    df = df.rename(columns=rename_map)

    critical_cols = [col for col in ['restaurant_name', 'city'] if col in df.columns]
    if critical_cols:
        df = df.dropna(subset=critical_cols)

    if 'city' in df.columns:
        df['city'] = df['city'].str.lower().str.strip()
        
    if 'cuisines' in df.columns:
        df['cuisines'] = df['cuisines'].str.lower().str.strip().fillna('unknown')
        
    cost_cols = df.filter(like='cost').columns
    cost_col = cost_cols[0] if not cost_cols.empty else None
    if cost_col:
        if df[cost_col].dtype == 'object':
            df[cost_col] = df[cost_col].astype(str).str.replace(',', '', regex=False)
            df[cost_col] = df[cost_col].str.extract(r'(\d+)').astype(float)

    rating_col = next((col for col in df.columns if 'rating' in col), None)
    if rating_col:
        if df[rating_col].dtype == 'object':
            df[rating_col] = df[rating_col].replace(['NEW', '-'], pd.NA)
            df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'cleaned_zomato_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Data saved to '{output_path}'")
    return df

if __name__ == "__main__":
    load_and_clean_data()
