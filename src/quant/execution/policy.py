from __future__ import annotations

from typing import Any


def portfolio_policy(config: dict[str, Any]) -> dict[str, Any]:
    policy = dict(config["portfolio_policy"])
    target = float(policy["target_position_weight"])
    gross = float(policy["max_gross_exposure"])
    cash = float(policy["minimum_cash_reserve"])
    sector = float(policy["max_sector_exposure"])
    max_positions = int(policy["max_positions"])
    candidate_count = int(policy["candidate_count"])
    if not 0 < target <= float(config["max_single_position"]):
        raise ValueError("Target position weight must be positive and within the single-name cap")
    if not 0 < gross < 1:
        raise ValueError("Maximum gross exposure must be between zero and one")
    if not 0 < cash < 1 or gross + cash > 1.0 + 1e-12:
        raise ValueError("Cash reserve is inconsistent with the gross exposure cap")
    if not target <= sector <= gross:
        raise ValueError("Sector exposure cap must cover one target position and stay within gross cap")
    if max_positions != int(gross / target + 1e-12):
        raise ValueError("max_positions must equal max_gross_exposure / target weight")
    if candidate_count < max_positions:
        raise ValueError("Candidate count cannot be below max_positions")
    return policy


def execution_signal(
    rank: int,
    current_weight: float,
    target_weight: float,
    selected_count: int,
) -> str:
    if target_weight <= 0:
        return "EXIT" if current_weight > 0 else "AVOID"
    if current_weight > target_weight + 1e-9:
        return "REDUCE"
    if abs(current_weight - target_weight) <= 1e-9:
        return "HOLD"
    strong_cutoff = max(1, min(5, selected_count // 4))
    return "STRONG_BUY" if rank <= strong_cutoff else "BUY"


def policy_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    policy = portfolio_policy(config)
    target = float(policy["target_position_weight"])
    return {
        "rule_version": str(config["rule_version"]),
        "candidate_count": int(policy["candidate_count"]),
        "candidate_refresh": str(policy["candidate_refresh"]),
        "target_position_weight": target,
        "initial_order_weight": target * float(config["initial_entry_fraction"]),
        "reduced_initial_order_weight": (
            target
            * float(config["initial_entry_fraction"])
            * float(config["calibration"]["reduced_entry_multiplier"])
        ),
        "add_order_weight": target * float(config["add_fraction"]),
        "max_gross_exposure": float(policy["max_gross_exposure"]),
        "minimum_cash_reserve": float(policy["minimum_cash_reserve"]),
        "max_sector_exposure": float(policy["max_sector_exposure"]),
        "max_single_position": float(config["max_single_position"]),
        "max_positions": int(policy["max_positions"]),
        "max_daily_new_positions": int(config["max_daily_new_positions"]),
        "simulation_only": bool(config["simulation_only"]),
    }
