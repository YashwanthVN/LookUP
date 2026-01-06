def safe_pct(value):
    return round(value * 100, 2) if value else None

def extract_financial_metrics(info):
    return {
        "market_cap": info.get("marketCap"),
        "revenue_growth_pct": safe_pct(info.get("revenueGrowth")),
        "earnings_growth_pct": safe_pct(info.get("earningsGrowth")),
        "profit_margin_pct": safe_pct(info.get("profitMargins")),
        "ebitda_margin_pct": safe_pct(info.get("ebitdaMargins")),
        "roe_pct": safe_pct(info.get("returnOnEquity")),
        "debt_to_equity": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "pe_ratio": info.get("trailingPE"),
        "price_to_book": info.get("priceToBook")
    }
