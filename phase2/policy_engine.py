def retention_policy(p_churn, risk_score, max_discount=10):
    seg = "High" if p_churn >= 0.7 else "Mid" if p_churn >= 0.4 else "Low"
    ladder = {
        "Low": ["values"],
        "Mid": ["values"],
        "High": ["values"]
    }[seg]
    # risk gate: reduce discount cap for high-risk underwriting
    cap = max_discount if risk_score <= 60 else max(0, max_discount-4)
    return {"segment": seg, "allowed_actions": ladder, "max_discount_pct": cap}
