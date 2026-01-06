import sys
import os
import streamlit as st
import yfinance as yf

# Allow src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_fetcher import fetch_company_data
from src.financial_metrics import extract_financial_metrics
from src.interpretation import interpret_metrics
from src.preprocessing import preprocess_financials

from src.peer_selector import get_sector_peers_finnhub
from src.peer_profiles import build_peer_profiles
from src.peer_ranking import rank_peers

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(page_title="LookUP", layout="wide")
st.title("📈 LookUP: AI Competitor Intelligence Platform")

# -------------------------------
# User Input
# -------------------------------
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA)").upper()

# -------------------------------
# Main Execution Block
# -------------------------------
if ticker:
    # ---- Fetch company data ----
    info, history = fetch_company_data(ticker)

    # ---- Preprocess financials ----
    processed_data = preprocess_financials(info)

    # ---- Extract & interpret metrics ----
    metrics = extract_financial_metrics(info)
    interpretation = interpret_metrics(metrics)

    # ---- Peer selection & ranking ----
    peer_data = get_sector_peers_finnhub(ticker)
    profiles = build_peer_profiles(peer_data["peers"])
    rankings = rank_peers(profiles)

    # -------------------------------
    # UI Rendering
    # -------------------------------
    st.subheader("Company Overview")

    st.subheader("Preprocessed Financial Metrics")
    st.json(processed_data)

    st.subheader("Key Financial Metrics")
    st.table(metrics)

    st.subheader("Financial Interpretation")
    st.json(interpretation)

    st.subheader(f"Industry: {peer_data['sector']}")
    st.json(rankings)

    st.subheader("Stock Price History (1 Year)")
    st.line_chart(history["Close"])
