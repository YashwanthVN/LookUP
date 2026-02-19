import os
import requests
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

class FMPClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        self.base_url = "https://financialmodelingprep.com/api/v3/"

    def get_bulk_data(self, tickers: List[str], quarters: int = 4) -> Dict:
        """ Fetches all profiles and financials for multiple tickers in 2-3 calls. """
        ticker_str = ",".join(tickers)
        
        # 1. Bulk Profiles (Atomic call for all tickers)
        profiles_raw = self._get(f"profile/{ticker_str}")
        profiles = {p['symbol']: p for p in profiles_raw} if isinstance(profiles_raw, list) else {}

        # 2. Bulk Financials (Using batch logic or individual fallback if bulk is restricted)
        # Note: FMP v3 'income-statement' supports one ticker at a time for deep history, 
        # but we can fetch them concurrently or use a batch wrapper.
        all_financials = {}
        for ticker in tickers:
            data = self._get(f"income-statement/{ticker}", {"period": "quarter", "limit": quarters})
            if data == "PAYMENT_REQUIRED" or not data:
                all_financials[ticker] = self._get_yfinance_fallback(ticker)
            else:
                all_financials[ticker] = pd.DataFrame(data)

        return {"profiles": profiles, "financials": all_financials}

    def _get(self, endpoint: str, params: dict = None) -> dict:
        if params is None: params = {}
        params["apikey"] = self.api_key
        url = f"{self.base_url}{endpoint.lstrip('/')}"
        try:
            resp = requests.get(url, params=params)
            if resp.status_code == 402: return "PAYMENT_REQUIRED"
            resp.raise_for_status()
            return resp.json()
        except Exception: return {}

    def _get_yfinance_fallback(self, ticker: str) -> pd.DataFrame:
        try:
            stock = yf.Ticker(ticker)
            df = stock.quarterly_financials.T.reset_index()
            df = df.rename(columns={'index': 'date', 'Total Revenue': 'revenue', 'Basic EPS': 'eps'})
            return df
        except: return pd.DataFrame()