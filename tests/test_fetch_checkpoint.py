from datetime import date

import pandas as pd

from quant.cli import (
    _fetch_checkpoint_id,
    _historical_symbol_fetch_range,
    _select_symbol_shard,
)


def test_fetch_checkpoint_is_deterministic_and_interval_scoped() -> None:
    first = _fetch_checkpoint_id("SH600000", date(2010, 1, 1), date(2026, 7, 27))
    repeated = _fetch_checkpoint_id("SH600000", date(2010, 1, 1), date(2026, 7, 27))
    another_end = _fetch_checkpoint_id("SH600000", date(2010, 1, 1), date(2026, 7, 28))

    assert first == repeated
    assert first != another_end


def test_symbol_shards_are_disjoint_and_complete() -> None:
    symbols = [f"SH{600000 + index:06d}" for index in range(11)]
    first = _select_symbol_shard(symbols, 0, 2)
    second = _select_symbol_shard(symbols, 1, 2)

    assert not set(first) & set(second)
    assert sorted(first + second) == symbols


def test_historical_constituent_fetch_continues_after_membership_exit() -> None:
    membership = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-06-28"]),
            "symbol": ["SH600000", "SH600000"],
        }
    )
    fetch_start, fetch_end = _historical_symbol_fetch_range(
        date(2024, 1, 1),
        date(2026, 7, 27),
        membership,
    )
    assert fetch_start == date(2024, 1, 1)
    assert fetch_end == date(2026, 7, 27)
