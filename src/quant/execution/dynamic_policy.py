from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _validate_config(config: dict[str, Any]) -> None:
    optimizer = config["optimizer"]
    gross = float(optimizer["maximum_gross_exposure"])
    if not 0 < gross <= 1.0:
        raise ValueError("Dynamic policy cannot borrow; maximum gross exposure must be in (0, 1]")
    if float(optimizer["base_risk_aversion"]) <= 0:
        raise ValueError("Base risk aversion must be positive")
    if float(optimizer["minimum_weight_change"]) < 0:
        raise ValueError("Minimum weight change cannot be negative")
    horizons = [int(value) for value in config["horizons"]]
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("Dynamic policy horizons must be positive")


def _horizon_value(row: pd.Series, stem: str, horizon: int, default: float) -> float:
    candidates = [f"{stem}_{horizon}d"]
    if horizon == 5:
        candidates.append(stem)
    for column in candidates:
        if column in row.index and pd.notna(row[column]):
            return _finite(row[column], default)
    return default


def evaluate_open_opportunity(
    row: pd.Series,
    relative_open_gap: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate only information fixed at T plus a T+1 official-open scenario."""
    _validate_config(config)
    optimizer = config["optimizer"]
    scenarios = config["open_scenarios"]
    costs = config["costs"]
    liquidity_penalty = np.clip(
        _finite(row.get("liquidity_penalty"), float(costs["liquidity_penalty_floor"])),
        float(costs["liquidity_penalty_floor"]),
        float(costs["liquidity_penalty_ceiling"]),
    )
    round_trip_cost = _finite(
        row.get("estimated_round_trip_rate"),
        float(costs["estimated_round_trip_rate"]),
    )
    is_incumbent = _finite(row.get("current_weight"), 0.0) > 0
    effective_cost = round_trip_cost * (
        float(config["holding_persistence"]["incumbent_remaining_cost_fraction"])
        if is_incumbent
        else 1.0
    )
    volatility_20 = max(
        _finite(row.get("volatility_20", row.get("vol_20")), 0.03),
        1e-4,
    )
    atr_fraction = np.clip(
        _finite(row.get("atr_fraction", row.get("atr_20_fraction")), volatility_20),
        float(scenarios["minimum_atr_fraction"]),
        float(scenarios["maximum_atr_fraction"]),
    )
    severe_adverse = (
        relative_open_gap
        <= -float(scenarios["severe_adverse_atr"]) * atr_fraction
    )

    alternatives: list[dict[str, float]] = []
    calibration_reliability = float(optimizer["calibration_reliability"])
    for horizon in [int(value) for value in config["horizons"]]:
        prediction = _horizon_value(row, "predicted_excess_return", horizon, np.nan)
        probability = np.clip(
            _horizon_value(row, "outperform_probability", horizon, 0.5),
            0.0,
            1.0,
        )
        if not math.isfinite(prediction):
            continue
        gap_sensitivity = float(
            scenarios["gap_sensitivity"].get(
                horizon,
                scenarios["gap_sensitivity"].get(str(horizon), 1.0),
            )
        )
        adjusted = prediction - gap_sensitivity * relative_open_gap
        net = adjusted - effective_cost - liquidity_penalty
        daily_net = net / horizon
        confidence = np.clip(
            max(0.0, 2.0 * (probability - 0.5)) * calibration_reliability,
            0.0,
            1.0,
        )
        horizon_risk = volatility_20 * math.sqrt(max(horizon, 1) / 20.0)
        uncertainty = horizon_risk * (1.0 - confidence)
        utility = (
            daily_net * (1.0 + float(optimizer["confidence_edge_multiplier"]) * confidence)
            - float(optimizer["uncertainty_penalty"]) * uncertainty / horizon
        )
        alternatives.append(
            {
                "horizon": float(horizon),
                "predicted_excess_return": prediction,
                "outperform_probability": probability,
                "confidence": float(confidence),
                "adjusted_excess_return": float(adjusted),
                "net_expected_return": float(net),
                "daily_net_edge": float(daily_net),
                "risk": float(horizon_risk),
                "utility": float(utility),
            }
        )
    if not alternatives:
        return {
            "eligible": False,
            "reason": "NO_VALID_MULTI_HORIZON_FORECAST",
            "selected_horizon": None,
            "confidence": 0.0,
            "daily_net_edge": float("-inf"),
            "utility": float("-inf"),
            "risk": volatility_20,
        }
    selected = max(
        alternatives,
        key=lambda value: (
            value["utility"],
            value["daily_net_edge"],
            -value["horizon"],
        ),
    )
    invalidated = bool(
        scenarios["invalidate_on_severe_adverse_gap"] and severe_adverse
    )
    eligible = (
        not invalidated
        and selected["daily_net_edge"]
        >= float(optimizer["minimum_daily_net_edge"])
    )
    reason = (
        "SEVERE_ADVERSE_OPEN_INVALIDATED"
        if invalidated
        else ("POSITIVE_NET_OPPORTUNITY" if eligible else "NET_EDGE_BELOW_THRESHOLD")
    )
    return {
        **{key: value for key, value in selected.items() if key != "horizon"},
        "selected_horizon": int(selected["horizon"]),
        "eligible": eligible,
        "reason": reason,
        "relative_open_gap": float(relative_open_gap),
        "atr_fraction": float(atr_fraction),
        "alternatives": alternatives,
    }


def _soft_target_weights(
    opportunities: pd.DataFrame,
    config: dict[str, Any],
) -> pd.Series:
    optimizer = config["optimizer"]
    eligible = opportunities["eligible"].fillna(False)
    raw = pd.Series(0.0, index=opportunities.index, dtype="float64")
    if not eligible.any():
        return raw
    view = opportunities.loc[eligible]
    confidence = view["confidence"].astype(float).clip(0.0, 1.0)
    edge = view["daily_net_edge"].astype(float).clip(lower=0.0)
    risk = view["risk"].astype(float).clip(lower=1e-4)
    risk_aversion = float(optimizer["base_risk_aversion"]) / (
        1.0 + float(optimizer["confidence_aggressiveness"]) * confidence.pow(2)
    )
    denominator = (
        risk_aversion * risk.pow(2)
        + float(optimizer["concentration_penalty"])
    )
    unconstrained = (
        edge
        * (1.0 + float(optimizer["confidence_edge_multiplier"]) * confidence)
        / denominator
    )

    # Sector exposure is a penalty, never a hard cap. An exceptional forecast may
    # therefore remain concentrated when its net utility dominates.
    if "sector_name" in view:
        sector_total = unconstrained.groupby(
            view["sector_name"].fillna("UNKNOWN")
        ).transform("sum")
        unconstrained = unconstrained / (
            1.0
            + float(optimizer["sector_concentration_penalty"])
            * sector_total
            / denominator
        )
    raw.loc[view.index] = unconstrained.clip(lower=0.0)
    gross = float(raw.sum())
    ceiling = float(optimizer["maximum_gross_exposure"])
    if gross > ceiling:
        raw *= ceiling / gross
    return raw


def allocate_dynamic_portfolio(
    candidates: pd.DataFrame,
    current_weights: dict[str, float],
    relative_open_gaps: dict[str, float],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Choose dynamic target weights with no fixed name/sector/cash quotas."""
    _validate_config(config)
    required = {"symbol"}
    if missing := required.difference(candidates.columns):
        raise ValueError(f"Dynamic candidates missing columns: {sorted(missing)}")
    frame = candidates.drop_duplicates("symbol").copy()
    frame["symbol"] = frame["symbol"].astype(str)
    absent_positions = sorted(set(current_weights).difference(frame["symbol"]))
    if absent_positions:
        frame = pd.concat(
            [
                frame,
                pd.DataFrame({"symbol": absent_positions}),
            ],
            ignore_index=True,
            sort=False,
        )
    frame["current_weight"] = frame["symbol"].map(current_weights).fillna(0.0)
    evaluated = [
        evaluate_open_opportunity(
            row,
            _finite(relative_open_gaps.get(str(row["symbol"])), 0.0),
            config,
        )
        for _, row in frame.iterrows()
    ]
    opportunity = pd.DataFrame(evaluated, index=frame.index)
    for column in opportunity.columns:
        frame[column] = opportunity[column]
    preselection = int(config["optimizer"]["candidate_preselection"])
    nonheld_eligible = frame["eligible"].fillna(False) & frame["current_weight"].le(0)
    eligible_rank = frame.loc[nonheld_eligible, "utility"].rank(
        method="first",
        ascending=False,
    )
    rejected = eligible_rank[eligible_rank > preselection].index
    if len(rejected):
        frame.loc[rejected, "eligible"] = False
        frame.loc[rejected, "reason"] = "OUTSIDE_DYNAMIC_PRESELECTION"

    incumbents = frame[
        frame["current_weight"].gt(0) & frame["eligible"].fillna(False)
    ]
    if not incumbents.empty:
        weakest_incumbent_utility = float(incumbents["utility"].min())
        replacement_multiple = float(
            config["holding_persistence"]["replacement_round_trip_cost_multiple"]
        )
        new_candidates = frame[
            frame["current_weight"].le(0) & frame["eligible"].fillna(False)
        ]
        for index, candidate in new_candidates.iterrows():
            horizon = max(1, int(candidate["selected_horizon"]))
            transaction_rate = _finite(
                candidate.get("estimated_round_trip_rate"),
                float(config["costs"]["estimated_round_trip_rate"]),
            )
            required_advantage = (
                transaction_rate * replacement_multiple / horizon
            )
            if (
                float(candidate["utility"])
                <= weakest_incumbent_utility + required_advantage
            ):
                frame.loc[index, "eligible"] = False
                frame.loc[index, "reason"] = "REPLACEMENT_EDGE_INSUFFICIENT"
    frame["unadjusted_target_weight"] = _soft_target_weights(frame, config)

    cost_multiplier = float(config["optimizer"]["turnover_cost_multiplier"])
    benefit_multiple = float(config["optimizer"]["no_trade_benefit_to_cost"])
    minimum_change = float(config["optimizer"]["minimum_weight_change"])
    targets: list[float] = []
    horizon_protected: list[bool] = []
    for _, row in frame.iterrows():
        current = max(0.0, float(row["current_weight"]))
        desired = max(0.0, float(row["unadjusted_target_weight"]))
        delta = desired - current
        transaction_rate = _finite(
            row.get("estimated_round_trip_rate"),
            float(config["costs"]["estimated_round_trip_rate"]),
        )
        expected_benefit = abs(delta) * abs(
            _finite(row.get("adjusted_excess_return"))
        )
        row_cost_multiplier = (
            float(config["optimizer"]["incumbent_no_trade_cost_multiplier"])
            if current > 0
            else cost_multiplier
        )
        expected_cost = abs(delta) * transaction_rate * row_cost_multiplier
        if abs(delta) < minimum_change or expected_benefit < expected_cost * benefit_multiple:
            desired = current
        if not bool(row["eligible"]) and current <= 0:
            desired = 0.0
        if not bool(row["eligible"]) and current > 0:
            # Invalid or negative signals are allowed to exit; they are never averaged down.
            desired = 0.0
        holding_days = _finite(row.get("holding_trading_days"), np.inf)
        incumbent_horizon = max(
            1.0,
            _finite(
                row.get("incumbent_horizon"),
                row.get("selected_horizon"),
            ),
        )
        protected = bool(
            current > 0
            and bool(row["eligible"])
            and float(row["daily_net_edge"])
            > float(config["holding_persistence"]["emergency_exit_daily_edge"])
            and holding_days
            < incumbent_horizon
            * float(
                config["holding_persistence"][
                    "minimum_horizon_fraction_before_optional_reduction"
                ]
            )
        )
        if protected and desired < current:
            desired = current
        targets.append(desired)
        horizon_protected.append(protected)
    frame["target_weight"] = targets
    frame["horizon_protected"] = horizon_protected

    gross = float(frame["target_weight"].sum())
    ceiling = float(config["optimizer"]["maximum_gross_exposure"])
    if gross > ceiling + 1e-12:
        protected_weight = float(
            frame.loc[frame["horizon_protected"], "target_weight"].sum()
        )
        if protected_weight >= ceiling:
            frame.loc[frame["horizon_protected"], "target_weight"] *= (
                ceiling / protected_weight
            )
            frame.loc[~frame["horizon_protected"], "target_weight"] = 0.0
        else:
            flexible = ~frame["horizon_protected"]
            flexible_weight = float(frame.loc[flexible, "target_weight"].sum())
            if flexible_weight > 0:
                frame.loc[flexible, "target_weight"] *= (
                    (ceiling - protected_weight) / flexible_weight
                )
    frame["delta_weight"] = frame["target_weight"] - frame["current_weight"]

    def action(row: pd.Series) -> str:
        current = float(row["current_weight"])
        target = float(row["target_weight"])
        delta = target - current
        if current > 0 and target <= 1e-12:
            return "EXIT"
        if current <= 1e-12 and target > 1e-12:
            return "OPEN"
        if delta >= minimum_change:
            return "ADD"
        if delta <= -minimum_change:
            return "REDUCE"
        if current > 0:
            return "HOLD"
        return "CANCEL"

    frame["action"] = frame.apply(action, axis=1)
    frame["cash_weight"] = 1.0 - float(frame["target_weight"].sum())
    frame["rule_version"] = str(config["rule_version"])
    return frame.sort_values(
        ["target_weight", "symbol"],
        ascending=[False, True],
    ).reset_index(drop=True)


def build_open_scenario_plan(
    candidates: pd.DataFrame,
    current_weights: dict[str, float],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Precompute T+1 official-open scenarios using only T-close information."""
    _validate_config(config)
    rows: list[dict[str, Any]] = []
    multipliers = [float(value) for value in config["open_scenarios"]["atr_multipliers"]]
    for _, candidate in candidates.drop_duplicates("symbol").iterrows():
        symbol = str(candidate["symbol"])
        reference_close = _finite(candidate.get("reference_close", candidate.get("close")))
        if reference_close <= 0:
            raise ValueError(f"{symbol} has no valid T-day reference close")
        atr_fraction = np.clip(
            _finite(
                candidate.get("atr_fraction", candidate.get("atr_20_fraction")),
                float(config["open_scenarios"]["minimum_atr_fraction"]),
            ),
            float(config["open_scenarios"]["minimum_atr_fraction"]),
            float(config["open_scenarios"]["maximum_atr_fraction"]),
        )
        standalone = pd.DataFrame([candidate])
        for scenario_index, multiplier in enumerate(multipliers):
            gap = multiplier * atr_fraction
            decision = allocate_dynamic_portfolio(
                standalone,
                {symbol: _finite(current_weights.get(symbol))},
                {symbol: gap},
                config,
            ).iloc[0]
            rows.append(
                {
                    "symbol": symbol,
                    "scenario_index": scenario_index,
                    "atr_multiplier": multiplier,
                    "relative_open_gap": gap,
                    "scenario_open_price": reference_close * (1.0 + gap),
                    "action": str(decision["action"]),
                    "indicative_target_weight": float(decision["target_weight"]),
                    "selected_horizon": decision["selected_horizon"],
                    "confidence": float(decision["confidence"]),
                    "expected_net_return": float(
                        decision.get("net_expected_return", np.nan)
                    ),
                    "reason": str(decision["reason"]),
                    "rule_version": str(config["rule_version"]),
                }
            )
    return pd.DataFrame(rows)


def select_precomputed_open_scenarios(
    plan: pd.DataFrame,
    official_open_gaps: dict[str, float],
) -> pd.DataFrame:
    """Select the nearest immutable T-day branch for each T+1 official open."""
    selected: list[pd.Series] = []
    for symbol, group in plan.groupby("symbol", sort=True):
        if symbol not in official_open_gaps:
            continue
        gap = float(official_open_gaps[symbol])
        distance = (group["relative_open_gap"].astype(float) - gap).abs()
        row = group.loc[distance.idxmin()].copy()
        row["official_open_gap"] = gap
        selected.append(row)
    return pd.DataFrame(selected).reset_index(drop=True)


def quantize_open_scenario_gaps(
    candidates: pd.DataFrame,
    official_open_gaps: dict[str, float],
    config: dict[str, Any],
) -> dict[str, float]:
    """Vectorized historical equivalent of selecting an immutable scenario row."""
    _validate_config(config)
    multipliers = np.asarray(
        config["open_scenarios"]["atr_multipliers"],
        dtype="float64",
    )
    if len(multipliers) == 0:
        raise ValueError("At least one open-scenario multiplier is required")
    selected: dict[str, float] = {}
    for _, candidate in candidates.drop_duplicates("symbol").iterrows():
        symbol = str(candidate["symbol"])
        if symbol not in official_open_gaps:
            continue
        atr_fraction = np.clip(
            _finite(
                candidate.get("atr_fraction", candidate.get("atr_20_fraction")),
                float(config["open_scenarios"]["minimum_atr_fraction"]),
            ),
            float(config["open_scenarios"]["minimum_atr_fraction"]),
            float(config["open_scenarios"]["maximum_atr_fraction"]),
        )
        scenario_gaps = multipliers * atr_fraction
        index = int(
            np.abs(scenario_gaps - float(official_open_gaps[symbol])).argmin()
        )
        selected[symbol] = float(scenario_gaps[index])
    return selected
