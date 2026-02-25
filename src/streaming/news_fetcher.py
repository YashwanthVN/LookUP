import yfinance as yf
import requests

class UnifiedNewsFetcher:
    def __init__(self, finnhub_api_key: str):
        self.finnhub_key = finnhub_api_key
        self.finnhub_url = "https://finnhub.io/api/v1/company-news"

    def fetch(self, ticker: str, limit: int = 5):
        # 1. Primary: Finnhub
        news = self._fetch_finnhub(ticker, limit)
        if news:
            print(f"✅ Retrieved {len(news)} items from Finnhub.")
            return news

        # 2. Secondary: yfinance
        print(f"⚠️ Finnhub failed or empty. Falling back to yfinance...")
        news = self._fetch_yfinance(ticker, limit)
        return news

    def _fetch_finnhub(self, ticker: str, limit: int):
        # Finnhub requires a date range; we'll look back 7 days
        from datetime import datetime, timedelta
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        params = {'symbol': ticker, 'from': start, 'to': end, 'token': self.finnhub_key}
        try:
            r = requests.get(self.finnhub_url, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                # Normalize to standard format: [{'headline': ..., 'summary': ...}]
                return [{'headline': n.get('headline'), 'summary': n.get('summary'), 'source': 'Finnhub'} 
                        for n in data[:limit]]
        except:
            return None
        return None

    def _fetch_yfinance(self, ticker: str, limit: int):
        try:
            yf_ticker = yf.Ticker(ticker)
            # yfinance news format is a list of dicts with 'title' and 'publisher'
            return [{'headline': n.get('title'), 'summary': n.get('publisher'), 'source': 'yfinance'} 
                    for n in yf_ticker.news[:limit]]
        except Exception as e:
            print(f"❌ Critical: Both news sources failed for {ticker}. Error: {e}")
            return []