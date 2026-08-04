from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.backtest.engine import BacktestResult
from quant.backtest.metrics import performance_metrics
from quant.backtest.rules import is_order_blocked_by_limit
from quant.config import ProjectPaths, load_config
from quant.execution.simulator import execution_fees
from quant.research.dynamic_open_backtest import AdjustedAccount, AdjustedLot
from quant.signals.archive import immutable_write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weekly_signal_dates(calendar: pd.DatetimeIndex) -> set[pd.Timestamp]:
    signals: set[pd.Timestamp] = set()
    for index, timestamp in enumerate(calendar[:-1]):
        current = pd.Timestamp(timestamp).isocalendar()
        following = pd.Timestamp(calendar[index + 1]).isocalendar()
        if (current.year, current.week) != (following.year, following.week):
            signals.add(pd.Timestamp(timestamp))
    return signals


def _benchmark_risk_state(
    bars: pd.DataFrame,
    benchmark_symbol: str,
) -> pd.DataFrame:
    benchmark = (
        bars[bars["symbol"].astype(str).eq(benchmark_symbol)]
        .sort_values("trade_date")
        .drop_duplicates("trade_date")
        .copy()
    )
    adjusted_return = benchmark["hfq_close"].pct_change(fill_method=None)
    benchmark["annualized_volatility_20"] = adjusted_return.rolling(20, min_periods=20).std() * np.sqrt(252.0)
    benchmark["ma120"] = (
        benchmark["hfq_close"]
        .rolling(
            120,
            min_periods=120,
        )
        .mean()
    )
    return benchmark[
        [
            "trade_date",
            "hfq_close",
            "annualized_volatility_20",
            "ma120",
        ]
    ]


def _stock_risk_context(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.sort_values(["symbol", "trade_date"]).copy()
    grouped = frame.groupby("symbol", sort=False)
    adjusted_return = grouped["hfq_close"].pct_change(fill_method=None)
    frame["annualized_volatility_20"] = adjusted_return.groupby(frame["symbol"], sort=False).rolling(
        20, min_periods=10
    ).std().reset_index(level=0, drop=True) * np.sqrt(252.0)
    previous_close = grouped["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr_fraction_20"] = true_range.groupby(frame["symbol"], sort=False).rolling(
        20, min_periods=10
    ).mean().reset_index(level=0, drop=True) / frame["close"].replace(0.0, np.nan)
    return frame


def _convert_position(
    account: AdjustedAccount,
    source_symbol: str,
    target_symbol: str,
    share_ratio: float,
    source_adjust_factor: float,
    target_adjust_factor: float,
    trading_date: date,
    fractional_share_price: float,
) -> dict[str, float | int | str]:
    if share_ratio <= 0:
        raise ValueError("Corporate-action share ratio must be positive")
    if source_adjust_factor <= 0 or target_adjust_factor <= 0:
        raise ValueError("Corporate-action adjustment factors must be positive")
    source_shares = account.raw_shares(source_symbol, source_adjust_factor)
    if source_shares <= 0:
        return {
            "source_symbol": source_symbol,
            "target_symbol": target_symbol,
            "source_shares": 0,
            "target_shares": 0,
            "fractional_target_shares": 0.0,
            "cash_compensation": 0.0,
        }
    source_lots = account.lots.get(source_symbol, [])
    inherited_acquired_on = min(
        (lot.acquired_on for lot in source_lots),
        default=trading_date,
    )
    exact_target_shares = source_shares * share_ratio
    target_shares = int(math.floor(exact_target_shares + 1e-9))
    fractional_target_shares = max(0.0, exact_target_shares - target_shares)
    cash_compensation = fractional_target_shares * max(
        0.0,
        fractional_share_price,
    )
    account.lots.pop(source_symbol, None)
    if target_shares > 0:
        account.lots.setdefault(target_symbol, []).append(
            AdjustedLot(
                units=target_shares / target_adjust_factor,
                acquired_on=inherited_acquired_on,
            )
        )
    account.cash += cash_compensation
    return {
        "source_symbol": source_symbol,
        "target_symbol": target_symbol,
        "source_shares": source_shares,
        "target_shares": target_shares,
        "fractional_target_shares": fractional_target_shares,
        "cash_compensation": cash_compensation,
    }


def _exposure(
    candidate: dict[str, Any],
    benchmark_row: pd.Series | None,
    market_state: str = "DATA_UNAVAILABLE",
) -> float:
    if str(candidate["exposure_mode"]) == "full":
        exposure = 1.0
    else:
        if benchmark_row is None:
            exposure = float(candidate["minimum_exposure"])
        else:
            volatility = float(benchmark_row["annualized_volatility_20"])
            if not np.isfinite(volatility) or volatility <= 0:
                exposure = float(candidate["minimum_exposure"])
            else:
                exposure = min(
                    1.0,
                    float(candidate["target_annualized_volatility"]) / volatility,
                )
                if (
                    pd.notna(benchmark_row["ma120"])
                    and float(benchmark_row["hfq_close"])
                    < float(benchmark_row["ma120"])
                ):
                    exposure *= float(
                        candidate["below_ma120_exposure_multiplier"]
                    )
        exposure = float(
            np.clip(
                exposure,
                float(candidate["minimum_exposure"]),
                1.0,
            )
        )
    multipliers = candidate.get("market_state_exposure_multipliers")
    if multipliers:
        exposure *= float(
            multipliers.get(
                str(market_state),
                multipliers.get("DATA_UNAVAILABLE", 1.0),
            )
        )
    return float(np.clip(exposure, 0.0, 1.0))


def _project_capped_weights(
    raw: pd.Series,
    exposure: float,
    sector_by_symbol: dict[str, str],
    maximum_name_weight: float,
    maximum_sector_weight: float | None,
) -> dict[str, float]:
    scores = raw.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scores = scores.clip(lower=0.0)
    if scores.empty or float(scores.sum()) <= 0 or exposure <= 0:
        return {str(symbol): 0.0 for symbol in scores.index}
    name_cap = max(0.0, min(float(maximum_name_weight), float(exposure)))
    sector_cap = (
        float(exposure)
        if maximum_sector_weight is None
        else max(0.0, min(float(maximum_sector_weight), float(exposure)))
    )
    weights = pd.Series(0.0, index=scores.index, dtype="float64")
    for _ in range(100):
        deficit = float(exposure) - float(weights.sum())
        if deficit <= 1e-12:
            break
        sector_totals = weights.groupby(
            pd.Series({symbol: sector_by_symbol.get(str(symbol), "UNKNOWN") for symbol in weights.index})
        ).sum()
        capacity = pd.Series(0.0, index=weights.index, dtype="float64")
        for symbol in weights.index:
            sector = sector_by_symbol.get(str(symbol), "UNKNOWN")
            capacity.loc[symbol] = max(
                0.0,
                min(
                    name_cap - float(weights.loc[symbol]),
                    sector_cap - float(sector_totals.get(sector, 0.0)),
                ),
            )
        active = capacity.gt(1e-12) & scores.gt(0)
        if not active.any():
            break
        increment = deficit * scores.loc[active] / float(scores.loc[active].sum())
        increment = np.minimum(increment, capacity.loc[active])
        sector_increment = increment.groupby(
            pd.Series({symbol: sector_by_symbol.get(str(symbol), "UNKNOWN") for symbol in increment.index})
        ).sum()
        for sector, amount in sector_increment.items():
            available = max(
                0.0,
                sector_cap - float(sector_totals.get(str(sector), 0.0)),
            )
            if float(amount) > available + 1e-12:
                members = [
                    symbol
                    for symbol in increment.index
                    if sector_by_symbol.get(str(symbol), "UNKNOWN") == str(sector)
                ]
                increment.loc[members] *= available / float(amount)
        accepted = float(increment.sum())
        if accepted <= 1e-12:
            break
        weights.loc[increment.index] += increment
    return {str(symbol): float(value) for symbol, value in weights.items()}


def _cap_direct_weights(
    desired: pd.Series,
    exposure: float,
    sector_by_symbol: dict[str, str],
    maximum_name_weight: float,
    maximum_sector_weight: float | None,
) -> dict[str, float]:
    weights = (
        desired.astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0, upper=float(maximum_name_weight))
    )
    sector_cap = (
        float(exposure)
        if maximum_sector_weight is None
        else min(float(maximum_sector_weight), float(exposure))
    )
    for sector in sorted(set(sector_by_symbol.values())):
        members = [
            symbol
            for symbol in weights.index
            if sector_by_symbol.get(str(symbol), "UNKNOWN") == sector
        ]
        sector_total = float(weights.loc[members].sum())
        if sector_total > sector_cap > 0:
            weights.loc[members] *= sector_cap / sector_total
    total = float(weights.sum())
    if total > float(exposure) > 0:
        weights *= float(exposure) / total
    return {
        str(symbol): float(value)
        for symbol, value in weights.items()
    }


def _target_weights(
    signals: pd.DataFrame,
    targets: list[str],
    exposure: float,
    candidate: dict[str, Any],
) -> dict[str, float]:
    if not targets or exposure <= 0:
        return {}
    selected = (
        signals[signals["symbol"].astype(str).isin(targets)]
        .drop_duplicates("symbol")
        .set_index("symbol")
        .reindex(targets)
    )
    sector_by_symbol = {
        str(symbol): (
            "UNKNOWN"
            if pd.isna(selected.loc[symbol].get("sector_name"))
            else str(selected.loc[symbol].get("sector_name"))
        )
        for symbol in selected.index
    }
    weight_mode = str(candidate.get("weight_mode", "equal"))
    if weight_mode == "equal":
        raw = pd.Series(1.0, index=selected.index)
    elif weight_mode in {
        "alpha_inverse_volatility",
        "calibrated_alpha_budget",
    }:
        volatility = pd.to_numeric(
            selected.get(
                "annualized_volatility_20",
                pd.Series(np.nan, index=selected.index),
            ),
            errors="coerce",
        )
        finite_volatility = volatility[
            np.isfinite(volatility) & volatility.gt(0)
        ]
        fallback_volatility = (
            float(finite_volatility.median())
            if len(finite_volatility)
            else float(candidate["volatility_floor"])
        )
        volatility = volatility.fillna(fallback_volatility).clip(
            lower=float(candidate["volatility_floor"])
        )
        alpha_column = str(
            candidate.get(
                "weight_alpha_column",
                "calibrated_lower_bound",
            )
        )
        if alpha_column not in selected:
            raise ValueError(
                f"Alpha weight column {alpha_column!r} is absent"
            )
        alpha = pd.to_numeric(
            selected[alpha_column],
            errors="coerce",
        ).fillna(0.0)
        alpha = (
            alpha - float(candidate.get("alpha_cost_hurdle", 0.0))
        ).clip(lower=0.0)
        if weight_mode == "calibrated_alpha_budget":
            full_size_alpha = float(candidate["alpha_full_size"])
            if full_size_alpha <= 0:
                raise ValueError("alpha_full_size must be positive")
            confidence = (alpha / full_size_alpha).clip(0.0, 1.0)
            risk_scale = (
                float(candidate.get("risk_volatility_target", 0.20))
                / volatility
            ).clip(upper=1.0)
            desired = (
                float(candidate.get("max_single_weight", exposure))
                * confidence.pow(float(candidate.get("alpha_power", 1.0)))
                * risk_scale.pow(
                    float(candidate.get("volatility_power", 1.0))
                )
            )
            return _cap_direct_weights(
                desired,
                exposure,
                sector_by_symbol,
                float(candidate.get("max_single_weight", exposure)),
                (
                    None
                    if candidate.get("max_sector_weight") is None
                    else float(candidate["max_sector_weight"])
                ),
            )
        raw = alpha.pow(float(candidate.get("alpha_power", 1.0))) / (
            volatility.pow(float(candidate.get("volatility_power", 1.0)))
        )
    else:
        volatility = pd.to_numeric(
            (
                selected["annualized_volatility_20"]
                if "annualized_volatility_20" in selected
                else pd.Series(np.nan, index=selected.index)
            ),
            errors="coerce",
        )
        finite_volatility = volatility[np.isfinite(volatility) & volatility.gt(0)]
        fallback_volatility = (
            float(finite_volatility.median())
            if len(finite_volatility)
            else float(candidate["volatility_floor"])
        )
        volatility = volatility.fillna(fallback_volatility).clip(lower=float(candidate["volatility_floor"]))
        score = pd.to_numeric(selected["model_score"], errors="coerce").fillna(0.0)
        probability = pd.to_numeric(
            (
                selected["outperform_probability"]
                if "outperform_probability" in selected
                else pd.Series(0.5, index=selected.index)
            ),
            errors="coerce",
        ).fillna(0.5)
        probability_confidence = (2.0 * (probability - 0.5)).clip(0.0, 1.0)
        conviction = (
            float(candidate["minimum_conviction"])
            + float(candidate["score_weight"]) * score.clip(0.0, 1.0)
            + float(candidate["probability_weight"]) * probability_confidence
        )
        raw = conviction.pow(float(candidate["conviction_power"])) / volatility.pow(
            float(candidate["volatility_power"])
        )
    sector_multipliers = candidate.get("sector_state_weight_multipliers")
    if sector_multipliers:
        sector_state = selected.get(
            "sector_state",
            pd.Series("DATA_UNAVAILABLE", index=selected.index),
        ).fillna("DATA_UNAVAILABLE")
        raw *= sector_state.map(sector_multipliers).fillna(
            float(sector_multipliers.get("DATA_UNAVAILABLE", 1.0))
        )
    return _project_capped_weights(
        raw,
        exposure,
        sector_by_symbol,
        float(candidate.get("max_single_weight", exposure)),
        (None if candidate.get("max_sector_weight") is None else float(candidate["max_sector_weight"])),
    )


def _market_state(signals: pd.DataFrame) -> str:
    if "market_state" not in signals or signals.empty:
        return "DATA_UNAVAILABLE"
    values = signals["market_state"].dropna().astype(str)
    if values.empty:
        return "DATA_UNAVAILABLE"
    return str(values.mode().iloc[0])


def _candidate_for_market_state(
    candidate: dict[str, Any],
    market_state: str,
) -> dict[str, Any]:
    dynamic_topk = candidate.get("topk_by_market_state")
    if not dynamic_topk:
        return candidate
    effective = dict(candidate)
    effective["topk"] = int(
        dynamic_topk.get(
            str(market_state),
            dynamic_topk.get("DATA_UNAVAILABLE", candidate["topk"]),
        )
    )
    effective["topk"] = max(1, min(effective["topk"], int(candidate["topk"])))
    return effective


def _target_symbols(
    signals: pd.DataFrame,
    held: set[str],
    candidate: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    ranking_column = str(candidate.get("ranking_column", "model_score"))
    if ranking_column not in signals:
        raise ValueError(f"Ranking column {ranking_column!r} is absent")
    all_ranked = signals.dropna(subset=[ranking_column]).sort_values(
        [ranking_column, "symbol"],
        ascending=[False, True],
    )
    all_ranked = all_ranked.drop_duplicates("symbol")
    all_rank = {
        str(symbol): index + 1
        for index, symbol in enumerate(
            all_ranked["symbol"].astype(str)
        )
    }
    ranked = all_ranked.copy()
    eligibility_column = candidate.get("eligibility_column")
    if eligibility_column is not None:
        eligibility_column = str(eligibility_column)
        if eligibility_column not in ranked:
            raise ValueError(
                f"Eligibility column {eligibility_column!r} is absent"
            )
        eligible_value = pd.to_numeric(
            ranked[eligibility_column],
            errors="coerce",
        )
        ranked = ranked[
            eligible_value.ge(
                float(candidate.get("minimum_eligibility_value", 0.0))
            )
        ]
    minimum_agreement = candidate.get("minimum_horizon_agreement")
    if minimum_agreement is not None:
        if "horizon_agreement" not in ranked:
            raise ValueError("Horizon agreement is absent")
        ranked = ranked[
            pd.to_numeric(
                ranked["horizon_agreement"],
                errors="coerce",
            ).ge(float(minimum_agreement))
        ]
    ranked = ranked.sort_values(
        [ranking_column, "symbol"],
        ascending=[False, True],
    )
    ranked = ranked.drop_duplicates("symbol")
    rank = {str(symbol): index + 1 for index, symbol in enumerate(ranked["symbol"].astype(str))}
    exit_rank = int(candidate["exit_rank"])
    max_drop = int(candidate["max_drop"])
    absent_from_signal = sorted(
        [symbol for symbol in held if symbol not in all_rank],
    )
    ineligible = sorted(
        [
            symbol
            for symbol in held
            if symbol in all_rank and symbol not in rank
        ],
        key=lambda symbol: (all_rank.get(symbol, math.inf), symbol),
        reverse=True,
    )
    ranked_exits = sorted(
        [symbol for symbol in held if symbol in rank and rank[symbol] > exit_rank],
        key=lambda symbol: (rank[symbol], symbol),
        reverse=True,
    )
    discretionary_exits = [
        *ineligible,
        *ranked_exits,
    ]
    exits = [
        *absent_from_signal,
        *discretionary_exits[:max_drop],
    ]
    survivors = held.difference(exits)

    sector_limit = candidate.get("sector_position_limit")
    sector_by_symbol = {
        str(row["symbol"]): ("UNKNOWN" if pd.isna(row.get("sector_name")) else str(row.get("sector_name")))
        for _, row in ranked.iterrows()
    }
    sector_count: dict[str, int] = {}
    for symbol in survivors:
        sector = sector_by_symbol.get(symbol, "UNKNOWN")
        sector_count[sector] = sector_count.get(sector, 0) + 1

    topk = int(candidate["topk"])
    maximum_buys = min(max_drop, max(0, topk - len(survivors)))
    buys: list[str] = []
    if maximum_buys <= 0:
        return sorted(survivors)[:topk], exits, buys
    for symbol in ranked["symbol"].astype(str):
        if symbol in survivors or symbol in exits:
            continue
        sector = sector_by_symbol.get(symbol, "UNKNOWN")
        if sector_limit is not None and sector_count.get(sector, 0) >= int(sector_limit):
            continue
        buys.append(symbol)
        sector_count[sector] = sector_count.get(sector, 0) + 1
        if len(buys) >= maximum_buys:
            break
    targets = sorted(survivors | set(buys))
    return targets[:topk], exits, buys


def _material_positions(
    account: AdjustedAccount,
    day: pd.DataFrame,
) -> set[str]:
    positions: set[str] = set()
    for symbol in account.lots:
        if account.units(symbol) <= 1e-10:
            continue
        if symbol not in day.index:
            # A temporarily missing or suspended holding must still reserve a
            # slot, otherwise a later reappearance can breach the name cap.
            positions.add(symbol)
            continue
        factor = float(day.loc[symbol, "adjust_factor"])
        # Corporate-action arithmetic can leave less than one raw share in
        # adjusted-unit space. Such a non-material residual cannot be traded
        # or reported as a holding and must not permanently consume a slot.
        if account.raw_shares(symbol, factor) > 0:
            positions.add(symbol)
    return positions


def _material_position_count(
    account: AdjustedAccount,
    day: pd.DataFrame,
) -> int:
    return len(_material_positions(account, day))


def _position_availability(
    account: AdjustedAccount,
    day: pd.DataFrame,
    trading_date: date,
    known_suspensions: set[tuple[str, date]],
) -> tuple[set[str], set[str]]:
    material = _material_positions(account, day)
    absent = {symbol for symbol in material if symbol not in day.index}
    explicit_suspensions = {
        symbol
        for symbol in material
        if symbol in day.index and not bool(day.loc[symbol, "is_trading"])
    }
    confirmed_absent_suspensions = {
        symbol
        for symbol in absent
        if (symbol, trading_date) in known_suspensions
    }
    suspended = explicit_suspensions | confirmed_absent_suspensions
    missing_data = absent - confirmed_absent_suspensions
    return suspended, missing_data


def _is_preexisting_position_cap_breach(
    before_new_entries: int,
    after_trades: int,
    maximum_positions: int,
) -> bool:
    if after_trades <= maximum_positions:
        return False
    if before_new_entries <= maximum_positions:
        raise RuntimeError(
            "Trading logic created a position cap breach: "
            f"before={before_new_entries}, after={after_trades}, maximum={maximum_positions}"
        )
    return True


def _purge_nonmaterial_residuals(
    account: AdjustedAccount,
    day: pd.DataFrame,
    trading_date: date,
) -> list[str]:
    purged: list[str] = []
    for symbol in list(account.lots):
        if symbol not in day.index or account.units(symbol) <= 1e-10:
            continue
        factor = float(day.loc[symbol, "adjust_factor"])
        if account.raw_shares(symbol, factor) > 0:
            continue
        lots = account.lots.get(symbol, [])
        if any(lot.acquired_on >= trading_date for lot in lots):
            continue
        # Selling an integer raw-share balance can leave less than one share
        # in adjusted-unit arithmetic. It is not a deliverable position and
        # must not reappear after a later adjustment-factor change.
        account.lots.pop(symbol, None)
        purged.append(symbol)
    return purged


@dataclass
class _RunningCalibration:
    count: int = 0
    total: float = 0.0
    total_square: float = 0.0
    wins: int = 0

    def add(self, value: float) -> None:
        if not np.isfinite(value):
            return
        self.count += 1
        self.total += float(value)
        self.total_square += float(value) ** 2
        self.wins += int(value > 0)

    def summary(self) -> dict[str, float]:
        if self.count <= 0:
            return {
                "count": 0.0,
                "mean": np.nan,
                "standard_deviation": np.nan,
                "win_rate": np.nan,
            }
        mean = self.total / self.count
        variance = max(0.0, self.total_square / self.count - mean**2)
        return {
            "count": float(self.count),
            "mean": float(mean),
            "standard_deviation": float(math.sqrt(variance)),
            "win_rate": float(self.wins / self.count),
        }


def _gap_bucket(value: float, edges: list[float]) -> int:
    if len(edges) < 2:
        raise ValueError("Open-gap calibration needs at least two edges")
    return int(np.digitize([float(value)], np.asarray(edges[1:-1]))[0])


class OpenCalibrationBook:
    def __init__(
        self,
        schedule: dict[pd.Timestamp, list[dict[str, Any]]],
        config: dict[str, Any],
    ) -> None:
        self.schedule = {
            pd.Timestamp(timestamp).normalize(): records for timestamp, records in schedule.items()
        }
        self.config = config
        self.pending_dates = sorted(self.schedule)
        self.pending_index = 0
        self.statistics: dict[tuple[str, str, int], _RunningCalibration] = {}

    def mature(self, current_date: pd.Timestamp) -> None:
        cutoff = pd.Timestamp(current_date).normalize()
        while (
            self.pending_index < len(self.pending_dates) and self.pending_dates[self.pending_index] < cutoff
        ):
            available_date = self.pending_dates[self.pending_index]
            for record in self.schedule[available_date]:
                market = str(record["market_state"])
                sector = str(record["sector_state"])
                bucket = int(record["gap_bucket"])
                outcome = float(record["realized_excess_return"])
                for key in (
                    (market, sector, bucket),
                    (market, "ANY", bucket),
                    ("ANY", "ANY", bucket),
                ):
                    self.statistics.setdefault(key, _RunningCalibration()).add(outcome)
            self.pending_index += 1

    def decision(
        self,
        market_state: str,
        sector_state: str,
        standardized_relative_gap: float,
    ) -> dict[str, Any]:
        bucket = _gap_bucket(
            standardized_relative_gap,
            [float(value) for value in self.config["gap_bucket_edges_atr"]],
        )
        candidates = [
            (
                "market_sector_state",
                (str(market_state), str(sector_state), bucket),
                int(self.config["minimum_exact_samples"]),
            ),
            (
                "market_state",
                (str(market_state), "ANY", bucket),
                int(self.config["minimum_market_samples"]),
            ),
            (
                "global",
                ("ANY", "ANY", bucket),
                int(self.config["minimum_global_samples"]),
            ),
        ]
        summary: dict[str, float] | None = None
        selected_level: str | None = None
        for level, key, minimum in candidates:
            stats = self.statistics.get(key)
            if stats is None or stats.count < minimum:
                continue
            summary = stats.summary()
            selected_level = level
            break
        if summary is None:
            return {
                "scale": float(self.config["insufficient_history_scale"]),
                "reason": "INSUFFICIENT_WALK_FORWARD_OPEN_HISTORY",
                "calibration_level": None,
                "sample_count": 0,
                "gap_bucket": bucket,
            }
        count = max(1, int(summary["count"]))
        standard_error = float(summary["standard_deviation"]) / math.sqrt(count)
        conservative_edge = (
            float(summary["mean"])
            - float(self.config["standard_error_multiplier"]) * standard_error
            - float(self.config["estimated_round_trip_rate"])
        )
        win_rate = float(summary["win_rate"])
        decision_mode = str(self.config.get("decision_mode", "hard_gate"))
        if decision_mode == "mild_cost_aware":
            mean_edge_after_cost = float(summary["mean"]) - float(
                self.config["estimated_round_trip_rate"]
            )
            if (
                mean_edge_after_cost <= float(self.config["hard_reject_mean_edge"])
                and win_rate <= float(self.config["hard_reject_max_win_rate"])
            ):
                scale = 0.0
                reason = "OPEN_BUCKET_STRONG_NEGATIVE_REJECTED"
            elif (
                conservative_edge <= float(self.config["soft_reduce_conservative_edge"])
                and win_rate <= float(self.config["soft_reduce_max_win_rate"])
            ):
                scale = float(self.config["soft_reduce_scale"])
                reason = "OPEN_BUCKET_MILD_REDUCTION"
            else:
                scale = 1.0
                reason = "OPEN_BUCKET_NO_MATERIAL_ADVERSE_EDGE"
        elif conservative_edge <= 0 or win_rate < float(self.config["minimum_win_rate"]):
            scale = 0.0
            reason = "OPEN_BUCKET_NET_EDGE_REJECTED"
        else:
            denominator = max(
                1e-9,
                float(self.config["full_size_win_rate"]) - 0.5,
            )
            reliability = np.clip((win_rate - 0.5) / denominator, 0.0, 1.0)
            scale = max(
                float(self.config["minimum_positive_scale"]),
                float(reliability),
            )
            reason = "OPEN_BUCKET_FULL_SIZE" if scale >= 1.0 - 1e-12 else "OPEN_BUCKET_REDUCED_SIZE"
        return {
            "scale": float(scale),
            "reason": reason,
            "calibration_level": selected_level,
            "sample_count": count,
            "gap_bucket": bucket,
            "mean_excess_return": float(summary["mean"]),
            "win_rate": win_rate,
            "conservative_edge_after_cost": float(conservative_edge),
        }


def _build_open_calibration_schedule(
    stocks: pd.DataFrame,
    benchmark: pd.DataFrame,
    predictions: pd.DataFrame,
    signal_dates: set[pd.Timestamp],
    config: dict[str, Any],
) -> dict[pd.Timestamp, list[dict[str, Any]]]:
    frame = stocks.sort_values(["symbol", "trade_date"]).copy()
    grouped = frame.groupby("symbol", sort=False)
    frame["next_raw_open"] = grouped["open"].shift(-1)
    frame["entry_hfq_open"] = grouped["hfq_open"].shift(-1)
    frame["exit_hfq_open"] = grouped["hfq_open"].shift(-6)
    frame["outcome_available_date"] = grouped["trade_date"].shift(-6)
    benchmark_frame = benchmark.sort_values("trade_date").copy()
    benchmark_frame["benchmark_next_raw_open"] = benchmark_frame["open"].shift(-1)
    benchmark_frame["benchmark_entry_hfq_open"] = benchmark_frame["hfq_open"].shift(-1)
    benchmark_frame["benchmark_exit_hfq_open"] = benchmark_frame["hfq_open"].shift(-6)
    benchmark_frame = benchmark_frame[
        [
            "trade_date",
            "close",
            "benchmark_next_raw_open",
            "benchmark_entry_hfq_open",
            "benchmark_exit_hfq_open",
        ]
    ].rename(columns={"close": "benchmark_reference_close"})
    observations = frame[
        [
            "symbol",
            "trade_date",
            "close",
            "atr_fraction_20",
            "next_raw_open",
            "entry_hfq_open",
            "exit_hfq_open",
            "outcome_available_date",
        ]
    ].merge(benchmark_frame, on="trade_date", how="left")
    scored = predictions[predictions["trade_date"].isin(signal_dates)].sort_values(
        ["trade_date", "model_score", "symbol"],
        ascending=[True, False, True],
    )
    scored = scored.groupby("trade_date", sort=False).head(int(config["calibration_universe_top_n"]))
    observations = scored[
        [
            "symbol",
            "trade_date",
            "market_state",
            "sector_state",
        ]
    ].merge(observations, on=["symbol", "trade_date"], how="inner")
    valid = observations[
        [
            "close",
            "atr_fraction_20",
            "next_raw_open",
            "entry_hfq_open",
            "exit_hfq_open",
            "benchmark_reference_close",
            "benchmark_next_raw_open",
            "benchmark_entry_hfq_open",
            "benchmark_exit_hfq_open",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    observations = observations[valid.notna().all(axis=1)].copy()
    observations = observations[
        observations["atr_fraction_20"].astype(float).gt(0)
        & observations["close"].astype(float).gt(0)
        & observations["benchmark_reference_close"].astype(float).gt(0)
        & observations["entry_hfq_open"].astype(float).gt(0)
        & observations["exit_hfq_open"].astype(float).gt(0)
        & observations["benchmark_entry_hfq_open"].astype(float).gt(0)
        & observations["benchmark_exit_hfq_open"].astype(float).gt(0)
        & observations["outcome_available_date"].notna()
    ]
    stock_gap = observations["next_raw_open"].astype(float) / observations["close"].astype(float) - 1.0
    benchmark_gap = (
        observations["benchmark_next_raw_open"].astype(float)
        / observations["benchmark_reference_close"].astype(float)
        - 1.0
    )
    observations["standardized_relative_gap"] = (stock_gap - benchmark_gap) / observations[
        "atr_fraction_20"
    ].astype(float)
    stock_return = (
        observations["exit_hfq_open"].astype(float) / observations["entry_hfq_open"].astype(float) - 1.0
    )
    benchmark_return = (
        observations["benchmark_exit_hfq_open"].astype(float)
        / observations["benchmark_entry_hfq_open"].astype(float)
        - 1.0
    )
    observations["realized_excess_return"] = stock_return - benchmark_return
    edges = [float(value) for value in config["gap_bucket_edges_atr"]]
    observations["gap_bucket"] = observations["standardized_relative_gap"].map(
        lambda value: _gap_bucket(float(value), edges)
    )
    schedule: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    for _, row in observations.iterrows():
        available = pd.Timestamp(row["outcome_available_date"]).normalize()
        schedule.setdefault(available, []).append(
            {
                "market_state": str(row.get("market_state", "DATA_UNAVAILABLE")),
                "sector_state": str(row.get("sector_state", "DATA_UNAVAILABLE")),
                "gap_bucket": int(row["gap_bucket"]),
                "realized_excess_return": float(row["realized_excess_return"]),
            }
        )
    return schedule


class RankBufferSimulator:
    def __init__(
        self,
        research_config: dict[str, Any],
        execution_config: dict[str, Any],
        candidate: dict[str, Any],
    ) -> None:
        self.research = research_config
        self.execution = execution_config
        self.candidate = candidate

    def run(
        self,
        bars: pd.DataFrame,
        predictions: pd.DataFrame,
        simulation_start: pd.Timestamp | None = None,
        known_suspensions: set[tuple[str, date]] | None = None,
    ) -> BacktestResult:
        frame = bars.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
        frame["symbol"] = frame["symbol"].astype(str)
        frame = frame.sort_values(["trade_date", "symbol"])
        scored = predictions.copy()
        scored["trade_date"] = pd.to_datetime(scored["trade_date"]).dt.normalize()
        scored["symbol"] = scored["symbol"].astype(str)
        benchmark_symbol = str(self.research["benchmark_symbol"])
        risk = _benchmark_risk_state(frame, benchmark_symbol).set_index("trade_date")
        benchmark = frame[frame["symbol"].eq(benchmark_symbol)].copy()
        stocks = _stock_risk_context(frame[~frame["symbol"].eq(benchmark_symbol)].copy())
        full_calendar = pd.DatetimeIndex(sorted(stocks["trade_date"].unique()))
        calibration_signal_dates = _weekly_signal_dates(full_calendar)
        normalized_start = (
            None if simulation_start is None else pd.Timestamp(simulation_start).normalize()
        )
        calendar = (
            full_calendar
            if normalized_start is None
            else full_calendar[full_calendar >= normalized_start]
        )
        signal_dates = _weekly_signal_dates(calendar)
        benchmark_gap = (
            benchmark.sort_values("trade_date").set_index("trade_date")["open"]
            / benchmark.sort_values("trade_date").set_index("trade_date")["close"].shift(1)
            - 1.0
        ).to_dict()
        calibration_book: OpenCalibrationBook | None = None
        if str(self.candidate.get("open_filter_mode", "none")) == "walk_forward_gap_calibration":
            schedule = _build_open_calibration_schedule(
                stocks,
                benchmark,
                scored,
                calibration_signal_dates,
                self.research["open_filter"],
            )
            calibration_book = OpenCalibrationBook(
                schedule,
                self.research["open_filter"],
            )
        bars_by_day = {
            pd.Timestamp(day): group.set_index("symbol")
            for day, group in stocks.groupby("trade_date", sort=True)
        }
        predictions_by_day = {
            pd.Timestamp(day): group for day, group in scored.groupby("trade_date", sort=True)
        }
        account = AdjustedAccount(float(self.research["initial_cash"]))
        previous_raw_close: dict[str, float] = {}
        previous_hfq_close: dict[str, float] = {}
        previous_volume: dict[str, float] = {}
        previous_adjust_factor: dict[str, float] = {}
        if normalized_start is not None:
            seed = (
                stocks[stocks["trade_date"].lt(normalized_start)]
                .sort_values(["symbol", "trade_date"])
                .groupby("symbol", sort=False)
                .tail(1)
            )
            previous_raw_close = {
                str(row["symbol"]): float(row["close"])
                for _, row in seed.iterrows()
                if pd.notna(row["close"]) and float(row["close"]) > 0
            }
            previous_hfq_close = {
                str(row["symbol"]): float(row["hfq_close"])
                for _, row in seed.iterrows()
                if pd.notna(row["hfq_close"]) and float(row["hfq_close"]) > 0
            }
            previous_volume = {
                str(row["symbol"]): float(row["volume"])
                for _, row in seed.iterrows()
                if pd.notna(row["volume"]) and float(row["volume"]) >= 0
            }
            previous_adjust_factor = {
                str(row["symbol"]): float(row["adjust_factor"])
                for _, row in seed.iterrows()
                if pd.notna(row["adjust_factor"]) and float(row["adjust_factor"]) > 0
            }
        pending: dict[str, Any] | None = None
        nav_records: list[dict[str, Any]] = []
        holding_records: list[dict[str, Any]] = []
        trade_records: list[dict[str, Any]] = []
        order_records: list[dict[str, Any]] = []
        corporate_actions = sorted(
            self.research.get("corporate_actions", []),
            key=lambda action: str(action["effective_date"]),
        )
        applied_corporate_actions: set[str] = set()
        terminal_events = sorted(
            self.research.get("terminal_events", []),
            key=lambda event: str(event["effective_date"]),
        )
        applied_terminal_events: set[str] = set()
        suspension_status = known_suspensions or set()

        for timestamp in calendar:
            timestamp = pd.Timestamp(timestamp)
            trading_date = timestamp.date()
            day = bars_by_day[timestamp]
            if calibration_book is not None:
                calibration_book.mature(timestamp)
            self._apply_terminal_events(
                account,
                previous_raw_close,
                previous_adjust_factor,
                trading_date,
                terminal_events,
                applied_terminal_events,
                order_records,
            )
            self._apply_corporate_actions(
                account,
                day,
                previous_adjust_factor,
                trading_date,
                corporate_actions,
                applied_corporate_actions,
                order_records,
            )
            if pending is not None:
                self._rebalance(
                    account,
                    day,
                    previous_raw_close,
                    previous_volume,
                    trading_date,
                    pending,
                    trade_records,
                    order_records,
                    float(benchmark_gap.get(timestamp, 0.0)),
                    calibration_book,
                )
                pending = None

            _purge_nonmaterial_residuals(account, day, trading_date)
            suspended_symbols, missing_data_symbols = _position_availability(
                account,
                day,
                trading_date,
                suspension_status,
            )
            market_value = 0.0
            for symbol in account.lots:
                price = (
                    float(day.loc[symbol, "hfq_close"])
                    if symbol in day.index and pd.notna(day.loc[symbol, "hfq_close"])
                    else previous_hfq_close.get(symbol, 0.0)
                )
                market_value += account.units(symbol) * price
            nav = account.cash + market_value
            nav_records.append(
                {
                    "trade_date": timestamp,
                    "cash": account.cash,
                    "market_value": market_value,
                    "nav": nav,
                    "gross_exposure": market_value / nav if nav > 0 else 0.0,
                    "reserved_positions": _material_position_count(account, day),
                    "position_cap_limit": int(self.research["maximum_material_positions"]),
                    "unavailable_positions": len(
                        suspended_symbols | missing_data_symbols
                    ),
                    "suspended_positions": len(suspended_symbols),
                    "missing_data_positions": len(missing_data_symbols),
                    "suspended_symbols": ",".join(sorted(suspended_symbols)),
                    "missing_data_symbols": ",".join(sorted(missing_data_symbols)),
                }
            )
            for symbol in sorted(account.lots):
                if symbol not in day.index:
                    continue
                factor = float(day.loc[symbol, "adjust_factor"])
                shares = account.raw_shares(symbol, factor)
                if shares <= 0:
                    continue
                value = account.units(symbol) * float(day.loc[symbol, "hfq_close"])
                holding_records.append(
                    {
                        "trade_date": timestamp,
                        "symbol": symbol,
                        "raw_shares": shares,
                        "market_value": value,
                    }
                )

            if timestamp in signal_dates and timestamp in predictions_by_day:
                signals = predictions_by_day[timestamp].merge(
                    day[
                        [
                            "annualized_volatility_20",
                            "atr_fraction_20",
                        ]
                    ].reset_index(),
                    on="symbol",
                    how="left",
                )
                current_market_state = _market_state(signals)
                effective_candidate = _candidate_for_market_state(
                    self.candidate,
                    current_market_state,
                )
                held = _material_positions(account, day)
                targets, exits, buys = _target_symbols(
                    signals,
                    held,
                    effective_candidate,
                )
                benchmark_row = risk.loc[timestamp] if timestamp in risk.index else None
                exposure = _exposure(
                    self.candidate,
                    benchmark_row,
                    current_market_state,
                )
                target_weights = _target_weights(
                    signals,
                    targets,
                    exposure,
                    self.candidate,
                )
                signal_context = (
                    signals[signals["symbol"].astype(str).isin(targets)]
                    .drop_duplicates("symbol")
                    .set_index("symbol")
                    .to_dict(orient="index")
                )
                pending = {
                    "signal_date": timestamp,
                    "targets": targets,
                    "exits": exits,
                    "buys": buys,
                    "target_weights": target_weights,
                    "signal_context": signal_context,
                    "exposure": exposure,
                }

            for symbol, row in day.iterrows():
                if pd.notna(row["close"]) and float(row["close"]) > 0:
                    previous_raw_close[str(symbol)] = float(row["close"])
                if pd.notna(row["hfq_close"]) and float(row["hfq_close"]) > 0:
                    previous_hfq_close[str(symbol)] = float(row["hfq_close"])
                if pd.notna(row["volume"]) and float(row["volume"]) >= 0:
                    previous_volume[str(symbol)] = float(row["volume"])
                if pd.notna(row["adjust_factor"]) and float(row["adjust_factor"]) > 0:
                    previous_adjust_factor[str(symbol)] = float(row["adjust_factor"])

        return BacktestResult(
            pd.DataFrame(nav_records),
            pd.DataFrame(holding_records),
            pd.DataFrame(trade_records),
            pd.DataFrame(order_records),
        )

    @staticmethod
    def _apply_terminal_events(
        account: AdjustedAccount,
        previous_raw_close: dict[str, float],
        previous_adjust_factor: dict[str, float],
        trading_date: date,
        events: list[dict[str, Any]],
        applied_event_ids: set[str],
        orders: list[dict[str, Any]],
    ) -> None:
        for event in events:
            event_id = str(event["event_id"])
            if event_id in applied_event_ids:
                continue
            if trading_date < pd.Timestamp(event["effective_date"]).date():
                continue
            symbol = str(event["symbol"])
            if account.units(symbol) <= 1e-12:
                applied_event_ids.add(event_id)
                continue
            factor = previous_adjust_factor.get(symbol)
            prior_close = previous_raw_close.get(symbol)
            if factor is None or factor <= 0 or prior_close is None or prior_close <= 0:
                continue
            shares = account.raw_shares(symbol, factor)
            recovery_rate = float(event["recovery_rate"])
            recovery_cash = shares * prior_close * recovery_rate
            account.lots.pop(symbol, None)
            account.cash += recovery_cash
            applied_event_ids.add(event_id)
            orders.append(
                {
                    "trade_date": trading_date,
                    "symbol": symbol,
                    "side": "WRITE_OFF",
                    "status": "FILLED",
                    "reason": str(event["reason"]),
                    "event_id": event_id,
                    "shares": shares,
                    "last_close": prior_close,
                    "recovery_rate": recovery_rate,
                    "recovery_cash": recovery_cash,
                    "reference_urls": list(event.get("reference_urls", [])),
                }
            )

    @staticmethod
    def _apply_corporate_actions(
        account: AdjustedAccount,
        day: pd.DataFrame,
        previous_adjust_factor: dict[str, float],
        trading_date: date,
        actions: list[dict[str, Any]],
        applied_action_ids: set[str],
        orders: list[dict[str, Any]],
    ) -> None:
        for action in actions:
            action_id = str(action["action_id"])
            if action_id in applied_action_ids:
                continue
            effective_date = pd.Timestamp(action["effective_date"]).date()
            if trading_date < effective_date:
                continue
            source = str(action["source_symbol"])
            target = str(action["target_symbol"])
            if account.units(source) <= 1e-12:
                applied_action_ids.add(action_id)
                continue
            source_factor = previous_adjust_factor.get(source)
            if (
                source_factor is None
                or source_factor <= 0
                or target not in day.index
                or pd.isna(day.loc[target, "adjust_factor"])
                or float(day.loc[target, "adjust_factor"]) <= 0
                or pd.isna(day.loc[target, "open"])
                or float(day.loc[target, "open"]) <= 0
            ):
                continue
            conversion = _convert_position(
                account=account,
                source_symbol=source,
                target_symbol=target,
                share_ratio=float(action["share_ratio"]),
                source_adjust_factor=float(source_factor),
                target_adjust_factor=float(day.loc[target, "adjust_factor"]),
                trading_date=trading_date,
                fractional_share_price=float(day.loc[target, "open"]),
            )
            applied_action_ids.add(action_id)
            orders.append(
                {
                    "trade_date": trading_date,
                    "symbol": source,
                    "side": "CONVERT",
                    "status": "FILLED",
                    "reason": "CORPORATE_ACTION_SHARE_CONVERSION",
                    "action_id": action_id,
                    "target_symbol": target,
                    "share_ratio": float(action["share_ratio"]),
                    "source_shares": int(conversion["source_shares"]),
                    "target_shares": int(conversion["target_shares"]),
                    "fractional_target_shares": float(
                        conversion["fractional_target_shares"]
                    ),
                    "cash_compensation": float(conversion["cash_compensation"]),
                    "reference_urls": list(action.get("reference_urls", [])),
                }
            )

    def _rebalance(
        self,
        account: AdjustedAccount,
        day: pd.DataFrame,
        previous_close: dict[str, float],
        previous_volume: dict[str, float],
        trading_date: date,
        pending: dict[str, Any],
        trades: list[dict[str, Any]],
        orders: list[dict[str, Any]],
        benchmark_open_gap: float,
        calibration_book: OpenCalibrationBook | None,
    ) -> None:
        open_value = account.cash
        for symbol in account.lots:
            price = (
                float(day.loc[symbol, "hfq_open"])
                if symbol in day.index and pd.notna(day.loc[symbol, "hfq_open"])
                else 0.0
            )
            open_value += account.units(symbol) * price
        if open_value <= 0:
            return
        target_set = set(pending["targets"])
        band = max(
            float(self.candidate["no_trade_weight_band"]),
            float(self.research["open_filter"]["estimated_round_trip_rate"])
            * float(self.candidate.get("no_trade_cost_multiple", 0.0)),
        )
        slippage = float(self.execution["slippage_rate"])
        lot = int(self.research["lot_size"])
        participation = float(self.execution["max_liquidity_participation"])

        desired: dict[str, int] = {}
        for symbol in target_set:
            if symbol not in day.index or pd.isna(day.loc[symbol, "open"]):
                continue
            row = day.loc[symbol]
            open_price = float(row["open"])
            factor = float(row["adjust_factor"])
            current_shares = account.raw_shares(symbol, factor)
            current_weight = current_shares * open_price / open_value
            target_weight = float(pending["target_weights"].get(symbol, 0.0))
            if symbol not in pending["buys"] and abs(current_weight - target_weight) <= band:
                desired[symbol] = current_shares
            else:
                desired[symbol] = int(open_value * target_weight / open_price // lot * lot)
            if calibration_book is not None and desired[symbol] > current_shares:
                context = pending["signal_context"].get(symbol, {})
                atr_fraction = float(context.get("atr_fraction_20", np.nan))
                relative_gap = (
                    open_price / previous_close.get(symbol, open_price) - 1.0
                ) - benchmark_open_gap
                standardized_gap = (
                    relative_gap / atr_fraction if np.isfinite(atr_fraction) and atr_fraction > 0 else 0.0
                )
                decision = calibration_book.decision(
                    str(context.get("market_state", "DATA_UNAVAILABLE")),
                    str(context.get("sector_state", "DATA_UNAVAILABLE")),
                    standardized_gap,
                )
                addition = desired[symbol] - current_shares
                desired[symbol] = current_shares + int(addition * float(decision["scale"]) // lot * lot)
                orders.append(
                    {
                        **self._order(
                            trading_date,
                            symbol,
                            "BUY",
                            (
                                "FILTER_ALLOWED"
                                if float(decision["scale"]) >= 1.0 - 1e-12
                                else ("FILTER_REDUCED" if float(decision["scale"]) > 0 else "FILTER_REJECTED")
                            ),
                            str(decision["reason"]),
                        ),
                        "open_filter_scale": float(decision["scale"]),
                        "open_filter_level": decision.get("calibration_level"),
                        "open_filter_samples": int(decision.get("sample_count", 0)),
                        "open_filter_mean_excess_return": decision.get("mean_excess_return"),
                        "open_filter_win_rate": decision.get("win_rate"),
                        "open_filter_conservative_edge": decision.get("conservative_edge_after_cost"),
                        "standardized_relative_open_gap": standardized_gap,
                    }
                )

        for symbol in sorted(set(account.lots) | target_set):
            if symbol not in day.index:
                continue
            row = day.loc[symbol]
            factor = float(row["adjust_factor"])
            current = account.raw_shares(symbol, factor)
            sell = max(0, current - desired.get(symbol, 0))
            if sell <= 0:
                continue
            sellable = account.sellable_raw_shares(symbol, factor, trading_date)
            quantity = min(sell, sellable)
            reason = self._blocked("SELL", symbol, row, previous_close, trading_date)
            if reason or quantity <= 0:
                orders.append(self._order(trading_date, symbol, "SELL", "REJECTED", reason or "T_PLUS_ONE"))
                continue
            fill = float(row["open"]) * (1.0 - slippage)
            gross = fill * quantity
            fees = execution_fees("SELL", gross, trading_date, self.execution)
            account.sell(symbol, quantity, factor, trading_date)
            account.cash += gross - fees
            trades.append(self._trade(trading_date, symbol, "SELL", quantity, fill, fees))

        material_count = _material_position_count(account, day)
        material_count_before_new_entries = material_count
        for symbol in pending["targets"]:
            if symbol not in day.index or symbol not in desired:
                continue
            row = day.loc[symbol]
            factor = float(row["adjust_factor"])
            current = account.raw_shares(symbol, factor)
            buy = max(0, desired[symbol] - current)
            if buy <= 0:
                continue
            if current <= 0 and material_count >= int(self.research["maximum_material_positions"]):
                orders.append(
                    self._order(
                        trading_date,
                        symbol,
                        "BUY",
                        "REJECTED",
                        "POSITION_CAP_AFTER_FAILED_SALE",
                    )
                )
                continue
            reason = self._blocked("BUY", symbol, row, previous_close, trading_date)
            if reason:
                orders.append(self._order(trading_date, symbol, "BUY", "REJECTED", reason))
                continue
            liquidity = int(previous_volume.get(symbol, 0.0) * participation // lot * lot)
            quantity = min(buy, liquidity)
            fill = float(row["open"]) * (1.0 + slippage)
            while quantity >= lot:
                gross = fill * quantity
                fees = execution_fees("BUY", gross, trading_date, self.execution)
                if gross + fees <= account.cash + 1e-9:
                    break
                quantity -= lot
            if quantity < lot:
                orders.append(self._order(trading_date, symbol, "BUY", "REJECTED", "CASH_OR_LIQUIDITY"))
                continue
            gross = fill * quantity
            fees = execution_fees("BUY", gross, trading_date, self.execution)
            account.cash -= gross + fees
            account.buy(symbol, quantity, factor, trading_date)
            if current <= 0:
                material_count += 1
            trades.append(self._trade(trading_date, symbol, "BUY", quantity, fill, fees))
        verified_count = _material_position_count(account, day)
        maximum_positions = int(self.research["maximum_material_positions"])
        if _is_preexisting_position_cap_breach(
            material_count_before_new_entries,
            verified_count,
            maximum_positions,
        ):
            orders.append(
                {
                    **self._order(
                        trading_date,
                        "*PORTFOLIO*",
                        "HOLD",
                        "CAP_BREACH_CARRIED",
                        "PREEXISTING_UNRESOLVED_POSITION_CAP",
                    ),
                    "position_count": verified_count,
                    "maximum_positions": maximum_positions,
                }
            )

    @staticmethod
    def _blocked(
        side: str,
        symbol: str,
        row: pd.Series,
        previous_close: dict[str, float],
        trading_date: date,
    ) -> str | None:
        if not bool(row["is_trading"]):
            return "SUSPENDED"
        price = float(row["open"])
        prior = previous_close.get(symbol)
        if not np.isfinite(price) or price <= 0:
            return "INVALID_OPEN"
        if prior is None or prior <= 0:
            return "MISSING_PREVIOUS_CLOSE"
        if is_order_blocked_by_limit(
            side,
            price,
            prior,
            symbol,
            trading_date,
            bool(row["is_st"]),
        ):
            return "LIMIT_UP" if side == "BUY" else "LIMIT_DOWN"
        return None

    @staticmethod
    def _trade(
        trading_date: date,
        symbol: str,
        side: str,
        shares: int,
        price: float,
        fees: float,
    ) -> dict[str, Any]:
        return {
            "trade_date": trading_date,
            "symbol": symbol,
            "side": side,
            "shares": shares,
            "price": price,
            "gross": price * shares,
            "fees": fees,
        }

    @staticmethod
    def _order(
        trading_date: date,
        symbol: str,
        side: str,
        status: str,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "trade_date": trading_date,
            "symbol": symbol,
            "side": side,
            "status": status,
            "reason": reason,
        }


def _period_metrics(
    result: BacktestResult,
    benchmark: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    nav = result.nav[pd.to_datetime(result.nav["trade_date"]).between(start, end)]
    trades = result.trades[pd.to_datetime(result.trades["trade_date"]).between(start, end)]
    benchmark_period = benchmark[pd.to_datetime(benchmark["trade_date"]).between(start, end)]
    metrics = performance_metrics(nav, benchmark_period, trades)
    duration_years = max(len(nav) / 252.0, 1 / 252.0)
    metrics["annual_turnover_gross"] = float(metrics.get("turnover_gross", 0.0) / duration_years)
    holdings = result.holdings[pd.to_datetime(result.holdings["trade_date"]).between(start, end)]
    counts = holdings.groupby("trade_date")["symbol"].nunique()
    metrics["average_holdings"] = float(counts.reindex(nav["trade_date"]).fillna(0).mean())
    metrics["maximum_holdings"] = int(counts.max()) if len(counts) else 0
    metrics["average_cash_ratio"] = float((nav["cash"] / nav["nav"]).mean())
    metrics["maximum_reserved_positions"] = (
        int(nav["reserved_positions"].max()) if "reserved_positions" in nav else metrics["maximum_holdings"]
    )
    metrics["position_cap_breach_days"] = (
        int(nav["reserved_positions"].gt(nav["position_cap_limit"]).sum())
        if {"reserved_positions", "position_cap_limit"}.issubset(nav.columns)
        else 0
    )
    metrics["maximum_unavailable_positions"] = (
        int(nav["unavailable_positions"].max()) if "unavailable_positions" in nav else 0
    )
    metrics["maximum_suspended_positions"] = (
        int(nav["suspended_positions"].max()) if "suspended_positions" in nav else 0
    )
    metrics["maximum_missing_data_positions"] = (
        int(nav["missing_data_positions"].max()) if "missing_data_positions" in nav else 0
    )
    orders = (
        result.orders[pd.to_datetime(result.orders["trade_date"]).between(start, end)]
        if not result.orders.empty and "trade_date" in result.orders
        else pd.DataFrame()
    )
    if not orders.empty and "status" in orders:
        filter_orders = orders[orders["status"].astype(str).str.startswith("FILTER_")]
        metrics["open_filter_decisions"] = {
            str(status): int(count) for status, count in filter_orders["status"].value_counts().items()
        }
        metrics["open_filter_average_scale"] = (
            float(filter_orders["open_filter_scale"].dropna().mean())
            if "open_filter_scale" in filter_orders and filter_orders["open_filter_scale"].notna().any()
            else np.nan
        )
    return metrics


def _run_locked_rank_buffer_evaluation(
    paths: ProjectPaths,
    research: dict[str, Any],
    config_name: str,
) -> dict[str, Any]:
    execution = load_config(paths, "execution")
    predictions_path = paths.reports / "research" / str(research["prediction_file"])
    bars_path = paths.curated / "bars.parquet"
    history_start = pd.Timestamp(research["history_start"])
    locked_start, locked_end = map(pd.Timestamp, research["locked_period"])
    if history_start >= locked_start:
        raise ValueError("history_start must precede the locked period")
    predictions = pd.read_parquet(
        predictions_path,
        filters=[("trade_date", ">=", history_start), ("trade_date", "<=", locked_end)],
    )
    bars = pd.read_parquet(
        bars_path,
        columns=[
            "symbol",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "is_trading",
            "is_st",
            "adjust_factor",
            "hfq_open",
            "hfq_close",
        ],
        filters=[
            ("trade_date", ">=", history_start - pd.Timedelta(days=200)),
            ("trade_date", "<=", locked_end),
        ],
    )
    if predictions.empty or bars.empty:
        raise RuntimeError("Locked evaluation inputs are empty")
    prediction_max = pd.to_datetime(predictions["trade_date"]).max().normalize()
    benchmark_symbol = str(research["benchmark_symbol"])
    benchmark = bars[bars["symbol"].astype(str).eq(benchmark_symbol)][
        ["trade_date", "close"]
    ].copy()
    benchmark_max = pd.to_datetime(benchmark["trade_date"]).max().normalize()
    if prediction_max < locked_end or benchmark_max < locked_end:
        raise RuntimeError(
            "Locked evaluation cutoff is incomplete: "
            f"predictions={prediction_max.date()}, benchmark={benchmark_max.date()}, "
            f"required={locked_end.date()}"
        )

    candidate = dict(research["candidate"])
    simulator = RankBufferSimulator(research, execution, candidate)
    result = simulator.run(bars, predictions, simulation_start=locked_start)
    metrics = _period_metrics(result, benchmark, locked_start, locked_end)
    rules = research["locked_acceptance"]
    checks = {
        "annualized_return": float(metrics["annualized_return"])
        >= float(rules["minimum_annualized_return"]),
        "sharpe_ratio": float(metrics["sharpe_ratio"])
        >= float(rules["minimum_sharpe_ratio"]),
        "max_drawdown": float(metrics["max_drawdown"])
        >= float(rules["maximum_drawdown_floor"]),
        "excess_cumulative_return": float(metrics["excess_cumulative_return"])
        >= float(rules["minimum_excess_cumulative_return"]),
        "annual_turnover_gross": float(metrics["annual_turnover_gross"])
        <= float(rules["maximum_annual_turnover_gross"]),
        "maximum_holdings": int(metrics["maximum_holdings"])
        <= int(rules["maximum_material_positions"]),
    }
    acceptance = {"passed": all(checks.values()), "checks": checks, "rules": rules}
    output_root = (
        paths.reports
        / "research"
        / str(research["artifact_directory"])
        / str(candidate["name"])
    )
    if output_root.exists():
        raise FileExistsError(f"{output_root} already exists; locked evaluation cannot be rerun")
    output_root.mkdir(parents=True)
    result.nav.to_parquet(output_root / "nav.parquet", index=False)
    result.holdings.to_parquet(output_root / "holdings.parquet", index=False)
    result.trades.to_parquet(output_root / "trades.parquet", index=False)
    result.orders.to_parquet(output_root / "orders.parquet", index=False)
    config_path = paths.configs / f"{config_name}.yaml"
    report = {
        "status": (
            "RANK_BUFFER_LOCKED_ACCEPTED"
            if acceptance["passed"]
            else "RANK_BUFFER_LOCKED_REJECTED"
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": str(research["run_id"]),
        "history_start": str(history_start.date()),
        "locked_period": [str(locked_start.date()), str(locked_end.date())],
        "candidate": candidate,
        "effective_no_trade_weight_band": max(
            float(candidate["no_trade_weight_band"]),
            float(research["open_filter"]["estimated_round_trip_rate"])
            * float(candidate.get("no_trade_cost_multiple", 0.0)),
        ),
        "open_filter": research["open_filter"],
        "locked_metrics": metrics,
        "locked_acceptance": acceptance,
        "locked_period_read": True,
        "methodology_guards": {
            "single_candidate_preregistered_before_locked_read": True,
            "empty_account_at_locked_period_start": True,
            "pre_locked_history_used_only_for_walk_forward_calibration": True,
            "open_filter_uses_only_outcomes_available_before_execution_open": True,
            "official_open_is_the_only_execution_price": True,
            "intraday_trigger_order_is_not_simulated": True,
            "locked_evaluation_cannot_overwrite_artifacts": True,
        },
        "input_versions": {
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "predictions": str(predictions_path),
            "predictions_sha256": _sha256(predictions_path),
            "bars": str(bars_path),
            "bars_sha256": _sha256(bars_path),
            "prediction_cutoff": str(prediction_max.date()),
            "benchmark_cutoff": str(benchmark_max.date()),
        },
        "simulation_only": True,
    }
    report_path = immutable_write_json(
        paths.reports / "research" / str(research["report_file"]),
        report,
    )
    return {**report, "report": str(report_path)}


def run_rank_buffer_research(
    paths: ProjectPaths,
    config_name: str = "rank_buffer_research",
) -> dict[str, Any]:
    research = load_config(paths, config_name)
    if "locked_period" in research:
        return _run_locked_rank_buffer_evaluation(paths, research, config_name)
    execution = load_config(paths, "execution")
    predictions_path = paths.reports / "research" / str(research["prediction_file"])
    bars_path = paths.curated / "bars.parquet"
    design_start, design_end = map(pd.Timestamp, research["design_period"])
    validation_start, validation_end = map(pd.Timestamp, research["validation_period"])
    start = min(design_start, validation_start)
    end = max(design_end, validation_end)
    predictions = pd.read_parquet(
        predictions_path,
        filters=[("trade_date", ">=", start), ("trade_date", "<=", end)],
    )
    bars = pd.read_parquet(
        bars_path,
        columns=[
            "symbol",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "is_trading",
            "is_st",
            "adjust_factor",
            "hfq_open",
            "hfq_close",
        ],
        filters=[
            ("trade_date", ">=", start - pd.Timedelta(days=200)),
            ("trade_date", "<=", end),
        ],
    )
    benchmark = bars[bars["symbol"].astype(str).eq(str(research["benchmark_symbol"]))][
        ["trade_date", "close"]
    ]
    design_results: dict[str, Any] = {}
    validation_results: dict[str, Any] = {}
    simulations: dict[str, BacktestResult] = {}
    eligible: list[str] = []
    selection = research["selection"]
    for candidate in research["candidates"]:
        name = str(candidate["name"])
        simulator = RankBufferSimulator(research, execution, candidate)
        result = simulator.run(bars, predictions)
        simulations[name] = result
        design = _period_metrics(
            result,
            benchmark,
            design_start,
            design_end,
        )
        validation = _period_metrics(
            result,
            benchmark,
            validation_start,
            validation_end,
        )
        design_results[name] = design
        validation_results[name] = validation
        if (
            float(design["max_drawdown"]) >= float(selection["maximum_drawdown_floor"])
            and float(design["annual_turnover_gross"])
            <= float(selection["maximum_annual_turnover_gross"])
            and int(design["maximum_holdings"])
            <= int(research["maximum_material_positions"])
        ):
            eligible.append(name)
    if not eligible:
        selected_name = None
    else:
        metric_names = [
            str(selection["primary_metric"]),
            *[str(value) for value in selection["tie_breakers"]],
        ]
        selected_name = max(
            eligible,
            key=lambda name: tuple(float(design_results[name][metric]) for metric in metric_names),
        )
    acceptance = {"passed": False, "checks": {}, "rules": research["validation_acceptance"]}
    if selected_name is not None:
        selected_validation = validation_results[selected_name]
        rules = research["validation_acceptance"]
        checks = {
            "annualized_return": float(selected_validation["annualized_return"])
            >= float(rules["minimum_annualized_return"]),
            "sharpe_ratio": float(selected_validation["sharpe_ratio"])
            >= float(rules["minimum_sharpe_ratio"]),
            "max_drawdown": float(selected_validation["max_drawdown"])
            >= float(rules["maximum_drawdown_floor"]),
            "excess_cumulative_return": float(selected_validation["excess_cumulative_return"])
            >= float(rules["minimum_excess_cumulative_return"]),
        }
        acceptance = {"passed": all(checks.values()), "checks": checks, "rules": rules}
        output_root = paths.reports / "research" / str(research["artifact_directory"]) / selected_name
        if output_root.exists():
            raise FileExistsError(f"{output_root} already exists; use a revision")
        output_root.mkdir(parents=True)
        selected = simulations[selected_name]
        selected.nav.to_parquet(output_root / "nav.parquet", index=False)
        selected.holdings.to_parquet(output_root / "holdings.parquet", index=False)
        selected.trades.to_parquet(output_root / "trades.parquet", index=False)
        selected.orders.to_parquet(output_root / "orders.parquet", index=False)
    report = {
        "status": (
            "RANK_BUFFER_VALIDATION_ACCEPTED" if acceptance["passed"] else "RANK_BUFFER_VALIDATION_REJECTED"
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "design_period": [str(design_start.date()), str(design_end.date())],
        "validation_period": [str(validation_start.date()), str(validation_end.date())],
        "strategy_family": (
            "Weekly Top-K Dropout with rank buffer, capped risk weights, "
            "and optional walk-forward official-open filter"
        ),
        "candidates": research["candidates"],
        "open_filter": research["open_filter"],
        "selection": selection,
        "design_results": design_results,
        "selected_candidate": selected_name,
        "validation_results": validation_results,
        "selected_validation_acceptance": acceptance,
        "locked_period_read": False,
        "methodology_guards": {
            "selection_uses_design_period_only": True,
            "validation_does_not_change_candidate_parameters": True,
            "open_filter_uses_only_outcomes_available_before_execution_open": True,
            "official_open_is_the_only_execution_price": True,
            "intraday_trigger_order_is_not_simulated": True,
        },
        "input_versions": {
            "predictions": str(predictions_path),
            "predictions_sha256": _sha256(predictions_path),
            "bars": str(bars_path),
            "bars_sha256": _sha256(bars_path),
        },
        "simulation_only": True,
    }
    report_path = immutable_write_json(
        paths.reports / "research" / str(research["report_file"]),
        report,
    )
    return {**report, "report": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", default="rank_buffer_research")
    args = parser.parse_args()
    result = run_rank_buffer_research(
        ProjectPaths(args.root.resolve()),
        str(args.config),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
