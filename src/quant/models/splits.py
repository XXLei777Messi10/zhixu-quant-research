from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _previous_trading_day(calendar: pd.DatetimeIndex, value: pd.Timestamp, offset: int) -> pd.Timestamp:
    # Leave exactly ``offset`` full trading dates between the two windows.
    position = calendar.searchsorted(value, side="left") - offset - 1
    if position < 0:
        raise ValueError("Not enough calendar history for embargo")
    return pd.Timestamp(calendar[position])


def quarterly_splits(
    calendar: pd.DatetimeIndex,
    first_oos: str = "2014-01-01",
    validation_months: int = 12,
    embargo_days: int = 5,
    minimum_training_years: int = 3,
) -> list[TimeSplit]:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(calendar).normalize().unique()))
    if dates.empty:
        return []
    quarters = pd.date_range(pd.Timestamp(first_oos), dates.max(), freq="QS")
    splits: list[TimeSplit] = []
    for quarter in quarters:
        test_dates = dates[(dates >= quarter) & (dates < quarter + pd.offsets.QuarterBegin(startingMonth=1))]
        if test_dates.empty:
            continue
        test_start, test_end = pd.Timestamp(test_dates.min()), pd.Timestamp(test_dates.max())
        try:
            valid_end = _previous_trading_day(dates, test_start, embargo_days)
        except ValueError:
            continue
        valid_floor = valid_end - pd.DateOffset(months=validation_months)
        valid_dates = dates[(dates >= valid_floor) & (dates <= valid_end)]
        if valid_dates.empty:
            continue
        valid_start = pd.Timestamp(valid_dates.min())
        try:
            train_end = _previous_trading_day(dates, valid_start, embargo_days)
        except ValueError:
            continue
        train_dates = dates[dates <= train_end]
        if train_dates.empty:
            continue
        if train_dates.max() < train_dates.min() + pd.DateOffset(years=minimum_training_years):
            continue
        splits.append(
            TimeSplit(
                pd.Timestamp(train_dates.min()),
                pd.Timestamp(train_dates.max()),
                valid_start,
                valid_end,
                test_start,
                test_end,
            )
        )
    return splits


def current_training_split(
    calendar: pd.DatetimeIndex,
    validation_months: int = 12,
    embargo_days: int = 5,
    minimum_training_years: int = 3,
) -> TimeSplit:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(calendar).normalize().unique()))
    if len(dates) < embargo_days * 2 + 30:
        raise ValueError("Not enough trading dates to create a training split")
    valid_end = dates[-(embargo_days + 2)]
    valid_floor = valid_end - pd.DateOffset(months=validation_months)
    valid_dates = dates[(dates >= valid_floor) & (dates <= valid_end)]
    valid_start = pd.Timestamp(valid_dates.min())
    train_end = _previous_trading_day(dates, valid_start, embargo_days)
    train_dates = dates[dates <= train_end]
    if train_dates.empty:
        raise ValueError("Not enough pre-validation history")
    if train_dates.max() < train_dates.min() + pd.DateOffset(years=minimum_training_years):
        raise ValueError(f"Training history is shorter than {minimum_training_years} years")
    return TimeSplit(
        train_start=pd.Timestamp(train_dates.min()),
        train_end=pd.Timestamp(train_dates.max()),
        valid_start=valid_start,
        valid_end=pd.Timestamp(valid_end),
        test_start=pd.Timestamp(dates[-1]),
        test_end=pd.Timestamp(dates[-1]),
    )


def assert_no_overlap(split: TimeSplit, embargo_days: int, calendar: pd.DatetimeIndex) -> None:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(calendar).normalize().unique()))
    train_pos = dates.get_loc(split.train_end)
    valid_pos = dates.get_loc(split.valid_start)
    valid_end_pos = dates.get_loc(split.valid_end)
    test_pos = dates.get_loc(split.test_start)
    if valid_pos - train_pos <= embargo_days:
        raise AssertionError("Train/validation embargo is too short")
    if test_pos - valid_end_pos <= embargo_days:
        raise AssertionError("Validation/test embargo is too short")
