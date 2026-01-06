# src/preprocessing.py

def safe_percentage(value):
    """
    Converts decimal values to percentage.
    Example: 0.253 -> 25.3%
    """
    if value is None:
        return None
    try:
        return round(value * 100, 2)
    except:
        return None


def safe_billion(value):
    """
    Converts large numbers to billions.
    Example: 456000000000 -> 456.0 B
    """
    if value is None:
        return None
    try:
        return round(value / 1_000_000_000, 2)
    except:
        return None


def safe_float(value):
    """
    Safely converts values to float.
    """
    if value is None:
        return None
    try:
        return round(float(value), 2)
    except:
        return None


def preprocess_financials(info: dict) -> dict:
    """
    Extracts, cleans, normalizes, and structures
    key financial metrics from Yahoo Finance info JSON.
    """

    processed_data = {
        "growth_metrics": {
            "revenue_growth_pct": safe_percentage(info.get("revenueGrowth")),
            "earnings_growth_pct": safe_percentage(info.get("earningsGrowth")),
        },

        "profitability_metrics": {
            "net_profit_margin_pct": safe_percentage(info.get("profitMargins")),
            "ebitda_margin_pct": safe_percentage(info.get("ebitdaMargins")),
            "return_on_equity_pct": safe_percentage(info.get("returnOnEquity")),
        },

        "financial_health_metrics": {
            "debt_to_equity": safe_float(info.get("debtToEquity")),
            "current_ratio": safe_float(info.get("currentRatio")),
        },

        "valuation_metrics": {
            "pe_ratio": safe_float(info.get("trailingPE")),
            "price_to_book": safe_float(info.get("priceToBook")),
        },

        "company_scale": {
            "market_cap_billion_usd": safe_billion(info.get("marketCap")),
        }
    }

    return processed_data
