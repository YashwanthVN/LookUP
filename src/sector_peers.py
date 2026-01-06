SECTOR_PEER_UNIVERSE = {
    "Technology": [
        "NVDA", "AMD", "INTC", "AVGO", "QCOM", "TSM", "AAPL", "MSFT"
    ],
    "Consumer Cyclical": [
        "TSLA", "GM", "F", "TM", "HMC", "BYDDF"
    ],
    "Communication Services": [
        "META", "GOOGL", "NFLX", "DIS"
    ]
}

def get_sector_peers(sector: str, limit: int = 6):
    peers = SECTOR_PEER_UNIVERSE.get(sector, [])
    return peers[:limit]
