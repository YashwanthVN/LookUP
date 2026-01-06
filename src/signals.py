def generate_peer_signals(rankings):
    signals = {}

    for category in rankings:
        signals[category] = {}

        for metric, ranked_list in rankings[category].items():
            if len(ranked_list) < 2:
                continue

            top = ranked_list[0]
            bottom = ranked_list[-1]

            signals[category][metric] = {
                "leader": top["ticker"],
                "laggard": bottom["ticker"]
            }

    return signals
