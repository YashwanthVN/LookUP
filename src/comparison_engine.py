import yfinance as yf
from src.preprocessing import preprocess_financials
from src.competitor_fetcher import get_competitors


def fetch_company_profile(ticker: str):
    stock = yf.Ticker(ticker)
    info = stock.info
    processed = preprocess_financials(info)
    return processed

def compare_metrics(target, competitors):
    comparison = {}

    for category in target:
        comparison[category] = {}

        for metric in target[category]:
            target_value = target[category][metric]
            peer_values = []

            for comp in competitors:
                comp_value = competitors[comp].get(category, {}).get(metric)
                if comp_value is not None:
                    peer_values.append(comp_value)

            if not peer_values or target_value is None:
                comparison[category][metric] = {
                    "target": target_value,
                    "peer_average": None,
                    "verdict": "insufficient_data"
                }
                continue

            peer_avg = round(sum(peer_values) / len(peer_values), 2)

            # Define comparison direction
            higher_is_better = metric not in ["debt_to_equity", "pe_ratio"]

            if higher_is_better:
                verdict = "outperforming" if target_value > peer_avg else "underperforming"
            else:
                verdict = "better" if target_value < peer_avg else "worse"

            comparison[category][metric] = {
                "target": target_value,
                "peer_average": peer_avg,
                "verdict": verdict
            }

    return comparison

def run_competitor_analysis(ticker: str):
    target_profile = fetch_company_profile(ticker)
    competitors = get_competitors(ticker)

    competitor_profiles = {}

    for comp in competitors:
        competitor_profiles[comp] = fetch_company_profile(comp)

    comparison_report = compare_metrics(target_profile, competitor_profiles)

    return {
        "target": ticker,
        "competitors": competitors,
        "comparison": comparison_report
    }
