import os
import finnhub
from dotenv import load_dotenv

load_dotenv()

def get_sector_peers_finnhub(ticker: str):
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise RuntimeError("Missing FINNHUB_API_KEY")

    client = finnhub.Client(api_key=api_key)

    profile = client.company_profile2(symbol=ticker)
    peers = client.company_peers(ticker)

    return {
        "sector": profile.get("finnhubIndustry", "Unknown"),
        "peers": [p for p in peers if p != ticker]
    }