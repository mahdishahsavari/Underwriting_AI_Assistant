#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_llm_input.py

Usage:
    python3 build_llm_input.py risk_output.json churn_output.json

Takes:
  - Phase 1 risk JSON  (with top-level "values" including risk_class_score, col_44 as coverage)
  - Phase 2 churn JSON (with top-level "values" including p_churn, churn_segment, churn_top_reasons)

Builds:
  - A final JSON with the "input_block" structure used for LLM SFT:
      {
        "input_block": {
          "customer_profile": {...},
          "underwriting": {...},
          "churn": {...},
          "constraints": {...}
        }
      }

And saves it to:
  llm_input_YYYYMMDD_HHMMSS.json
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, date


# ---------------------------------------------------------------------
# Helpers for safe date parsing
# ---------------------------------------------------------------------
def _parse_date_safe(value):
    """Parse 'YYYY-MM-DD' into datetime.date, or return None."""
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.date() if isinstance(value, datetime) else value
    try:
        # Expecting an ISO-like string, e.g. "2023-05-10"
        return date.fromisoformat(str(value))
    except Exception:
        return None


def compute_customer_profile(churn_values: dict) -> dict:
    """
    Build the customer_profile dict from churn JSON "values".
    Expects keys:
      Years_With_Company, Total_Premium_Paid, Renewal_Count,
      Avg_Claim_Amount, Claim_Count,
      First_Policy_Date_Gre, Last_Renewal_Date_Gre, Last_Expiration_Date
    """
    years_with_company = churn_values.get("Years_With_Company")
    total_premium_paid = churn_values.get("Total_Premium_Paid")
    renewal_count = churn_values.get("Renewal_Count")
    avg_claim_amount = churn_values.get("Avg_Claim_Amount")
    claim_count = churn_values.get("Claim_Count")

    first_policy_date = _parse_date_safe(
        churn_values.get("First_Policy_Date_Gre"))
    last_renewal_date = _parse_date_safe(
        churn_values.get("Last_Renewal_Date_Gre"))
    last_expiration = _parse_date_safe(
        churn_values.get("Last_Expiration_Date"))

    # Reference date: last_expiration if present, else today
    ref_date = last_expiration or date.today()

    def _days_between(a, b):
        if a is None or b is None:
            return None
        return (a - b).days

    days_since_first_policy = _days_between(ref_date, first_policy_date)
    days_since_last_renewal = _days_between(ref_date, last_renewal_date)
    days_to_expiration = _days_between(
        last_expiration, ref_date) if last_expiration else None

    return {
        "years_with_company": years_with_company,
        "total_premium_paid": total_premium_paid,
        "renewal_count":      renewal_count,
        "avg_claim_amount":   avg_claim_amount,
        "claim_count":        claim_count,
        "days_since_first_policy": days_since_first_policy,
        "days_since_last_renewal": days_since_last_renewal,
        "days_to_expiration":      days_to_expiration,
    }


def map_risk_level(risk_score):
    """
    Map numeric risk_score (risk_class_score) to a Farsi risk_level string.

    Example rule:
      1-3  -> "Low"
      4-7  -> "Medium"
      8-10 -> "High"

    You can adjust this logic if your business rule is different.
    """
    if risk_score is None:
        return None
    try:
        s = float(risk_score)
    except Exception:
        return None

    if s <= 3:
        return "Low"
    elif s <= 7:
        return "Medium"
    else:
        return "High"


def compute_underwriting(risk_values: dict) -> dict:
    """
    Build the underwriting dict from risk JSON "values" (Phase 1 output).

    Expects:
      - risk_class_score : numeric class (e.g. 1..10)
      - col_44           : coverage text (e.g.)

    Output:
      {
        "risk_score":  <risk_class_score>,
        "risk_level":  <mapped low/medium/high>,
        "coverage":    <col_44>
      }
    """
    # 1) risk_score from ML
    risk_score = risk_values.get("risk_class_score")

    # 2) risk_level derived from score
    risk_level = map_risk_level(risk_score)

    # 3) coverage taken from col_44 in ML JSON
    coverage = risk_values.get("col_44")

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "coverage":   coverage,
    }


def compute_churn_section(churn_values: dict) -> dict:
    """
    Build the churn dict from churn JSON "values".
    Expects:
      - p_churn
      - churn_segment
      - optionally: churn_top_reasons (list or comma-separated string)
    """
    p_churn = churn_values.get("p_churn", 0.0)
    segment = churn_values.get("churn_segment", "")
    top_reasons = churn_values.get("churn_top_reasons", [])

    # Normalize top_reasons to list[str]
    if isinstance(top_reasons, str):
        if top_reasons.strip():
            top_reasons = [x.strip() for x in top_reasons.split(",")]
        else:
            top_reasons = []
    elif not isinstance(top_reasons, list):
        top_reasons = []

    try:
        p_churn = float(p_churn)
    except Exception:
        p_churn = 0.0

    return {
        "p_churn": p_churn,
        "segment": segment,
        "top_reasons": top_reasons,
    }


def build_constraints() -> dict:
    """
    Fixed constraints block, same idea as SFT dataset.
    """
    return {
        "max_discount_pct": 10,
        "allowed_actions": [
            "Allowed Values"
        ],
    }


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Build final LLM input JSON (input_block) from risk + churn JSON outputs."
    )
    ap.add_argument(
        "risk_json",  help="Path to Phase 1 risk JSON (with 'values').")
    ap.add_argument(
        "churn_json", help="Path to Phase 2 churn JSON (with 'values').")
    args = ap.parse_args()

    risk_path = Path(args.risk_json)
    churn_path = Path(args.churn_json)

    if not risk_path.is_file():
        raise FileNotFoundError(f"Risk JSON not found: {risk_path}")
    if not churn_path.is_file():
        raise FileNotFoundError(f"Churn JSON not found: {churn_path}")

    # --- Load JSONs ---
    with risk_path.open("r", encoding="utf-8") as f:
        risk_data = json.load(f)
    with churn_path.open("r", encoding="utf-8") as f:
        churn_data = json.load(f)

    if "values" not in risk_data or not isinstance(risk_data["values"], dict):
        raise ValueError(
            "Risk JSON must have a top-level 'values' object (dict).")
    if "values" not in churn_data or not isinstance(churn_data["values"], dict):
        raise ValueError(
            "Churn JSON must have a top-level 'values' object (dict).")

    risk_values = risk_data["values"]
    churn_values = churn_data["values"]

    # --- Build each section ---
    customer_profile = compute_customer_profile(churn_values)
    underwriting = compute_underwriting(risk_values)
    churn_section = compute_churn_section(churn_values)
    constraints = build_constraints()

    input_block = {
        "customer_profile": customer_profile,
        "underwriting":     underwriting,
        "churn":            churn_section,
        "constraints":      constraints,
    }

    output = {
        "input_block": input_block
    }

    # --- Save to file: llm_input_YYYYMMDD_HHMMSS.json ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"llm_input_{timestamp}.json"
    out_path = Path.cwd() / out_filename

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved LLM input JSON to: {out_path}")


if __name__ == "__main__":
    main()


"""
HINT: python3 build_llm_input.py risk_output.json churn_output.json

HOW IN INFERENCE OF LLM WILL WORK:

import json

with open("llm_input_....json", "r", encoding="utf-8") as f:
    data = json.load(f)

input_block = data["input_block"]
user_content = json.dumps(input_block, ensure_ascii=False)

messages = [
    {
        "role": "system",
        "content": (
            "You are an advanced fire insurance underwriting and retention assistant. "
            "Return ONLY valid JSON with keys: summary, retention_plan, underwriting_notes, cx_message_short."
        ),
    },
    {
        "role": "user",
        "content": user_content,
    },
]

# بعدش همون apply_chat_template + model.generate که قبلاً نوشتیم

"""
