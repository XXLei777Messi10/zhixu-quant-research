from __future__ import annotations

from typing import Any

import pandas as pd


def calibrate_from_oos_signals(
    bars: pd.DataFrame,
    predictions: pd.DataFrame,
    as_of: pd.Timestamp,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Estimate boundary parameters from completed historical OOS signal outcomes."""
    required_bars = {
        "symbol",
        "trade_date",
        "hfq_open",
        "hfq_high",
        "hfq_low",
        "hfq_close",
    }
    required_predictions = {"symbol", "trade_date", "model_score"}
    if missing := required_bars - set(bars):
        raise ValueError(f"Calibration bars missing: {sorted(missing)}")
    if missing := required_predictions - set(predictions):
        raise ValueError(f"Calibration predictions missing: {sorted(missing)}")

    settings = config["calibration"]
    as_of = pd.Timestamp(as_of).normalize()
    lookback_start = as_of - pd.DateOffset(years=int(settings["lookback_years"]))
    frame = bars.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame[
        frame["trade_date"].between(lookback_start - pd.Timedelta(days=40), as_of)
    ].sort_values(["symbol", "trade_date"])
    grouped = frame.groupby("symbol", sort=False)
    previous_close = grouped["hfq_close"].shift(1)
    true_range = pd.concat(
        [
            frame["hfq_high"] - frame["hfq_low"],
            (frame["hfq_high"] - previous_close).abs(),
            (frame["hfq_low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr20"] = (
        true_range.groupby(frame["symbol"])
        .rolling(20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    frame["next_open"] = grouped["hfq_open"].shift(-1)
    future_highs = pd.concat(
        [grouped["hfq_high"].shift(-offset) for offset in range(1, 6)],
        axis=1,
    )
    future_lows = pd.concat(
        [grouped["hfq_low"].shift(-offset) for offset in range(1, 6)],
        axis=1,
    )
    frame["future_high_5d"] = future_highs.max(axis=1, skipna=False)
    frame["future_low_5d"] = future_lows.min(axis=1, skipna=False)
    outcomes = frame[
        frame["trade_date"].ge(lookback_start)
        & frame[["atr20", "next_open", "future_high_5d", "future_low_5d"]]
        .notna()
        .all(axis=1)
    ].copy()
    outcomes["open_gap_atr"] = (
        outcomes["next_open"] - outcomes["hfq_close"]
    ) / outcomes["atr20"]
    outcomes["mfe_fraction"] = (
        outcomes["future_high_5d"] / outcomes["hfq_close"] - 1.0
    ).clip(lower=0.0)
    outcomes["mae_fraction"] = (
        1.0 - outcomes["future_low_5d"] / outcomes["hfq_close"]
    ).clip(lower=0.0)
    outcomes["mae_atr"] = (
        outcomes["hfq_close"] - outcomes["future_low_5d"]
    ).clip(lower=0.0) / outcomes["atr20"]

    signal_frame = predictions.copy()
    signal_frame["trade_date"] = pd.to_datetime(signal_frame["trade_date"])
    signal_frame = signal_frame[
        signal_frame["trade_date"].between(lookback_start, as_of)
        & signal_frame["model_score"].ge(float(settings["minimum_oos_model_score"]))
    ][["symbol", "trade_date", "model_score"]]
    samples = signal_frame.merge(
        outcomes[
            [
                "symbol",
                "trade_date",
                "open_gap_atr",
                "mfe_fraction",
                "mae_fraction",
                "mae_atr",
            ]
        ],
        on=["symbol", "trade_date"],
        how="inner",
        validate="one_to_one",
    )
    sample_count = len(samples)
    minimum = int(settings["minimum_peer_samples"])
    payload: dict[str, Any] = {
        "as_of": as_of.date().isoformat(),
        "lookback_start": lookback_start.date().isoformat(),
        "sample_count": sample_count,
        "minimum_peer_samples": minimum,
        "minimum_oos_model_score": float(settings["minimum_oos_model_score"]),
        "outcome_horizon_trading_days": 5,
        "uses_only_completed_oos_signals": True,
    }
    if sample_count < minimum:
        return {**payload, "calibration_status": "INSUFFICIENT_OOS_SAMPLES"}

    q = settings["quantiles"]
    gap_low = float(samples["open_gap_atr"].quantile(float(q["entry_low"])))
    gap_high = float(samples["open_gap_atr"].quantile(float(q["entry_high"])))
    gap_chase = float(samples["open_gap_atr"].quantile(float(q["chase"])))
    mae_add = float(samples["mae_atr"].quantile(float(q["add"])))
    mae_hard = float(samples["mae_atr"].quantile(float(q["hard_exit"])))
    source = (
        f"walk_forward_oos_{lookback_start.date().isoformat()}_"
        f"{as_of.date().isoformat()}_n{sample_count}"
    )
    parameters = {
        "entry_low_atr": max(0.01, -gap_low),
        "entry_high_atr": max(0.01, gap_high),
        "add_atr": max(0.01, mae_add),
        "chase_atr": max(0.02, gap_chase, gap_high + 0.01),
        "hard_exit_atr": max(0.02, mae_hard),
        "peer_mfe_q70": float(samples["mfe_fraction"].quantile(float(q["favorable"]))),
        "peer_mfe_q90": float(
            samples["mfe_fraction"].quantile(float(q["favorable_high"]))
        ),
        "peer_mae_q80": float(samples["mae_fraction"].quantile(float(q["adverse"]))),
    }
    return {
        **payload,
        "calibration_status": "WALK_FORWARD_PASS",
        "calibration_source": source,
        "parameters": parameters,
        "quantiles": q,
    }


def apply_calibration(
    contexts: dict[str, dict[str, Any]],
    calibration: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if calibration.get("calibration_status") != "WALK_FORWARD_PASS":
        return contexts
    parameters = dict(calibration["parameters"])
    return {
        symbol: {
            **context,
            **parameters,
            "calibration_status": "WALK_FORWARD_PASS",
            "calibration_source": calibration["calibration_source"],
        }
        for symbol, context in contexts.items()
    }
