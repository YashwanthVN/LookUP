def rank_peers(peer_profiles):
    rankings = {}

    for ticker, profile in peer_profiles.items():
        if not profile:
            continue

        for category, metrics in profile.items():
            rankings.setdefault(category, {})

            for metric, value in metrics.items():
                if value is None:
                    continue

                rankings[category].setdefault(metric, [])
                rankings[category][metric].append({
                    "ticker": ticker,
                    "value": value
                })

    # Sort rankings
    for category in rankings:
        for metric in rankings[category]:
            reverse = metric not in ["debt_to_equity", "pe_ratio"]
            rankings[category][metric] = sorted(
                rankings[category][metric],
                key=lambda x: x["value"],
                reverse=reverse
            )

    return rankings
