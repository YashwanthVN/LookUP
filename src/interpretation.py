def interpret_metrics(metrics):
    interpretations = {}

    if metrics["revenue_growth_pct"]:
        interpretations["growth"] = (
            "High Growth" if metrics["revenue_growth_pct"] > 15 else "Moderate/Low Growth"
        )

    if metrics["pe_ratio"]:
        interpretations["valuation"] = (
            "Overvalued" if metrics["pe_ratio"] > 30 else "Fairly Valued"
        )

    if metrics["debt_to_equity"]:
        interpretations["risk"] = (
            "High Risk" if metrics["debt_to_equity"] > 150 else "Stable"
        )

    return interpretations
