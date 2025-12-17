import yfinance as yf

def fetch_company_data(ticker):
    print(f"[LookUP] Fetching financial data for: {ticker}")
    stock = yf.Ticker(ticker)
    info = stock.info
    history = stock.history(period="1y")
    return info, history
