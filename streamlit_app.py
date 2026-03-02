import re
import streamlit as st
import os
import torch
from dotenv import load_dotenv
from src.graph.financial_kg import DynamicFinancialKG
from src.inference.reasoning_engine import LookUPReporter

# Page Config
st.set_page_config(page_title="LookUP | AI Summary", page_icon="📈", layout="wide")
load_dotenv()

# --- CACHE THE MODEL (Prevent reloading on every click) ---
@st.cache_resource
def load_engine():
    kg = DynamicFinancialKG()
    model_path = "calibrated_gnn_reasoning.pt"
    # Note: If you want to use it for commodities, we map query terms to tickers
    return kg, model_path

kg, model_path = load_engine()

# --- UI HEADER ---
st.title("📈 LookUP: Financial Reasoning")
st.markdown("### Explain market movements using GNN-Attention & Gemini")

# --- SEARCH BAR ---
query = st.text_input("Ask about a stock or commodity:", placeholder="Why has the price of gold fallen?")

# Mapping Logic for Commodities
ticker_map = {
    "sensex": "^BSESN",
    "bse": "^BSESN",
    "nifty": "^NSEI",
    "gold": "XAUUSD",
    "silver": "XAGUSD",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL"
}

def extract_ticker(query):
    # Remove punctuation and convert to lowercase
    clean = re.sub(r'[^\w\s]', '', query.lower())
    words = clean.split()
    
    # Check each word against the map
    for word in words:
        if word in ticker_map:
            return ticker_map[word]
    
    # If no match, maybe the user typed a ticker directly (e.g., "AAPL")
    # Check if the cleaned query itself might be a ticker (uppercase, 1-5 letters)
    potential_ticker = clean.upper().strip()
    if re.match(r'^[A-Z]{1,5}$', potential_ticker):
        return potential_ticker
    
    # Otherwise, return None (invalid)
    return None


if st.button("Analyze Causal Drivers"):
    if query:
        target_ticker = extract_ticker(query)
        
        if target_ticker is None:
            st.warning(f"Could not identify a valid ticker in your query. Please use a known name (e.g., 'sensex', 'gold', 'AAPL') or type a ticker directly.")
        else:
            with st.spinner(f"Building Knowledge Graph for {target_ticker}..."):
                try:
                    # 1. Build and Inject
                    kg.build_for_tickers([target_ticker])
                    kg.inject_real_time_news(target_ticker)
                    
                    # 2. Get Reasoning
                    reporter = LookUPReporter(kg, model_path)
                    top_drivers = reporter.engine.get_causal_drivers(target_ticker)
                    
                    # --- UI LAYOUT ---
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.subheader("GNN Attention Mapping [Extracting relevant news]")
                        if top_drivers:
                            for d in top_drivers:
                                with st.expander(f"Score: {d['impact_score']:.4f} | {d['headline'][:50]}..."):
                                    st.write(f"**Full Headline:** {d['headline']}")
                                    st.write(f"**GNN Weight:** {d['impact_score']:.4f}")
                                    st.write(f"**Sentiment:** {d['sentiment']:.2f}")
                                    st.progress(min(d['impact_score'] * 2, 1.0))
                        else:
                            st.warning("No significant news drivers found.")

                    with col2:
                        st.subheader("AI Summary Analysis")
                        report = reporter.generate_report(target_ticker)
                        st.info(report)
                        
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
    else:
        st.warning("Please enter a query or ticker first.")

# --- FOOTER ---
st.divider()
st.caption("LookUP v1.0 | Built with PyTorch Geometric, NetworkX, and Gemini Pro")