import yfinance as yf

def get_sector_industry(ticker: str):
    info = yf.Ticker(ticker).info
    return {
        "sector": info.get("sector"),
        "industry": info.get("industry")
    }
