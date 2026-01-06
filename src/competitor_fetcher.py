import yfinance as yf

MANUAL_PEERS = {
    "NVDA": ["AMD", "INTC", "AVGO", "QCOM", "TSM"],
    "AAPL": ["MSFT", "GOOGL", "META"],
    "TSLA": ["GM", "F", "BYDDF"]
}

def get_competitors(ticker: str, limit: int = 5):
    stock = yf.Ticker(ticker)

    try:
        peers = stock.get_peers()
        if peers:
            return peers[:limit]
    except:
        pass

    return MANUAL_PEERS.get(ticker.upper(), [])

