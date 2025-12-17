import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.data_fetcher import fetch_company_data

st.set_page_config(page_title="LookUP", layout="wide")
st.title("📈 LookUP: AI Competitor Intelligence Platform")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA)")

if ticker:
    info, history = fetch_company_data(ticker)
    st.subheader("Company Overview")
    st.json(info)

    st.subheader("Stock Price History (1 Year)")
    st.line_chart(history['Close'])
