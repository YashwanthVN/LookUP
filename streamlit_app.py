import re
import os
import torch
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

# Internal module imports
from src.graph.financial_kg import DynamicFinancialKG
from src.inference.reasoning_engine import LookUPReporter

# --- Page Config ---
st.set_page_config(
    page_title="LookUP | AI Summary", 
    page_icon="📈", 
    layout="wide"
)
load_dotenv()

# --- CACHE THE MODEL (Prevent reloading on every click) ---
@st.cache_resource
def load_engine():
    """
    Initializes the Knowledge Graph and defines the model path.
    Mapping query terms to tickers allows for commodity support.
    """
    kg = DynamicFinancialKG()
    model_path = "calibrated_gnn_reasoning.pt"
    return kg, model_path

kg, model_path = load_engine()

# --- UI HEADER ---
st.title("📈 LookUP: Financial Reasoning")
st.markdown("### Explain market movements using GNN-Attention & Gemini")

# --- SEARCH BAR ---
query = st.text_input(
    "Ask about a stock or commodity:", 
    placeholder="Why has the price of gold fallen?"
)

def resolve_ticker(query):
    """
    Dynamically resolves a stock name or commodity to a Yahoo Finance Ticker.
    Uses yfinance Search to avoid hardcoding.
    """
    stop_words = [
        "why", "is", "has", "the", "price", "of", "fallen",
        "risen", "today", "on", "what", "how", "situation", "in"
    ]
    
    # Clean the query to get just the subject
    query_words = [w for w in query.lower().split() if w not in stop_words]
    clean_subject = " ".join(query_words)
    
    try:
        search = yf.Search(clean_subject, max_results=1)
        if search.quotes:
            # The first result is usually the most relevant (e.g., "Apple" -> "AAPL")
            best_match = search.quotes[0]
            ticker = best_match['symbol']
            name = best_match.get('shortname', best_match.get('longname', ticker))
            return ticker, name
    except Exception as e:
        print(f"Ticker resolution failed: {e}")
    
    return None, None

# --- EXECUTION LOGIC ---
if st.button("Analyze Causal Drivers"):
    if query:
        target_ticker, display_name = resolve_ticker(query)
        
        if not target_ticker:
            st.warning(
                "Could not identify the stock or commodity. Try being more "
                "specific (e.g., 'Apple Inc' instead of just 'Apple')."
            )
        else:
            st.success(f"Resolved '{query}' to **{display_name}** ({target_ticker})")
            
            with st.spinner(f"Building Knowledge Graph for {target_ticker}..."):
                try:
                    # 1. Build KG and Inject News
                    kg.build_for_tickers([target_ticker])
                    kg.inject_real_time_news(target_ticker)
                    
                    # 2. Get Reasoning Engine
                    reporter = LookUPReporter(kg, model_path)
                    top_drivers = reporter.engine.get_causal_drivers(target_ticker)
                    
                    # --- UI LAYOUT ---
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.subheader("GNN Attention Mapping")
                        st.caption("Extracting high-influence news nodes")
                        
                        if top_drivers:
                            for d in top_drivers:
                                label = f"Score: {d['impact_score']:.4f} | {d['headline'][:50]}..."
                                with st.expander(label):
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