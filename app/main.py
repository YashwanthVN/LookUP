import sys, os
import yfinance as yf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.data_fetcher import fetch_company_data
from src.financial_metrics import extract_financial_metrics
from src.interpretation import interpret_metrics
from src.preprocessing import preprocess_financials

st.set_page_config(page_title="LookUP", layout="wide")
st.title("📈 LookUP: AI Competitor Intelligence Platform")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA)")
info = yf.Ticker(ticker).info
processed_data = preprocess_financials(info)

if ticker:
    info, history = fetch_company_data(ticker)
    st.subheader("Company Overview")
    metrics = extract_financial_metrics(info)
    interpretation = interpret_metrics(metrics)
    
    st.subheader("Preprocessed Financial Metrics")
    st.json(processed_data)

    st.subheader("Key Financial Metrics")
    st.table(metrics)

    st.subheader("Financial Interpretation")
    st.json(interpretation)


    st.subheader("Stock Price History (1 Year)")
    st.line_chart(history['Close'])
