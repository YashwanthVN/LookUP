import yfinance as yf
from src.preprocessing import preprocess_financials

def build_peer_profiles(peer_list):
    profiles = {}

    for ticker in peer_list:
        try:
            info = yf.Ticker(ticker).info
            profiles[ticker] = preprocess_financials(info)
        except:
            profiles[ticker] = None

    return profiles
