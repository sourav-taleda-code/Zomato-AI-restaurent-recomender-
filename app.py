import streamlit as st
import pandas as pd
import os
from data_ingestion import load_and_clean_data
from recommender import load_data, recommend, get_ai_recommendation

# Set up page configurations
st.set_page_config(page_title="Zomato AI Recommendation System", page_icon="🍕", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size: 2.8rem; font-weight: 700; color: #E03546; margin-bottom: 2rem; }
    .subheader { font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Zomato AI Recommendation System 🍕</div>', unsafe_allow_html=True)

csv_path = 'data/cleaned_zomato_data.csv'

# Step 1: Ingest Data check
if not os.path.exists(csv_path):
    st.info("Cleaned dataset not found. Please click below to download and clean the Zomato dataset from Hugging Face.")
    if st.button("Ingest Zomato Dataset", type="primary"):
        with st.spinner("Downloading and processing data (this may take a few seconds)..."):
            df = load_and_clean_data()
            if df is not None:
                st.success("Data successfully ingested!")
                st.rerun()
else:
    df = load_data(csv_path)
    
    # Sidebar - Step 2: User Inputs
    st.sidebar.header("Search Filters")
    cities = sorted(df['city'].dropna().unique())
    selected_city = st.sidebar.selectbox("Select Neighborhood/Area", cities, index=cities.index('btm') if 'btm' in cities else 0)
    
    max_price = st.sidebar.number_input("Max Budget (for two people)", min_value=0, value=1000, step=100)
    
    if st.sidebar.button("Get Recommendations", type="primary"):
        # Step 3 & 4: Integrate and Recommend
        results = recommend(df, selected_city, max_price)
        
        if results.empty:
            st.warning(f"No restaurants found in {selected_city.title()} under budget of ₹{max_price}.")
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div class="subheader">Top Ranked Restaurants</div>', unsafe_allow_html=True)
                show_df = results[['restaurant_name', 'cuisines', 'average_cost_for_two', 'rating', 'votes']].copy()
                show_df.columns = ['Name', 'Cuisines', 'Avg Cost for Two (₹)', 'Rating', 'Votes']
                st.dataframe(show_df, use_container_width=True, hide_index=True)
                
            with col2:
                st.markdown('<div class="subheader">AI Recommendation Insights (Groq)</div>', unsafe_allow_html=True)
                with st.spinner("Generating AI analysis..."):
                    restaurants_list = results.to_dict(orient='records')
                    ai_summary = get_ai_recommendation(restaurants_list, selected_city, max_price)
                    st.info(ai_summary)
