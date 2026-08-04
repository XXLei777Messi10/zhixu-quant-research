from datetime import UTC, datetime

import pandas as pd
import pytest

from quant.data.baostock_adapter import BaostockAdapter
from quant.data.normalize import (
    canonical_symbol,
    normalize_akshare_bars,
    normalize_akshare_tx_bars,
    normalize_baostock_bars,
)


def test_symbol_formats() -> None:
    assert canonical_symbol("600000") == "SH600000"
    assert canonical_symbol("sz.000001") == "SZ000001"
    assert canonical_symbol("830001") == "BJ830001"


def test_akshare_mapping_and_units() -> None:
    raw = pd.DataFrame(
        {
            "日期": ["2026-01-05"],
            "开盘": [10.0],
            "最高": [11.0],
            "最低": [9.0],
            "收盘": [10.5],
            "成交量": [123.0],
            "成交额": [129150.0],
            "换手率": [1.2],
        }
    )
    result = normalize_akshare_bars(raw, "600000", datetime.now(UTC), "raw", "req")
    assert result.loc[0, "volume"] == 12_300
    assert result.loc[0, "amount"] == 129_150
    assert result.loc[0, "turnover"] == pytest.approx(0.012)
    assert bool(result.loc[0, "is_trading"])


def test_akshare_tencent_mapping_and_units() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2026-01-05"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [123.0],
            "amount": [129150.0],
            "turnover": [0.012],
        }
    )
    result = normalize_akshare_tx_bars(raw, "600000", datetime.now(UTC), "raw", "req")
    assert result.loc[0, "volume"] == 12_300
    assert result.loc[0, "turnover"] == pytest.approx(0.012)


def test_akshare_tencent_shanghai_volume_already_in_shares() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2026-01-05"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [12_300.0],
            "amount": [129_150.0],
            "turnover": [0.012],
        }
    )
    result = normalize_akshare_tx_bars(raw, "600000", datetime.now(UTC), "raw", "req")
    assert result.loc[0, "volume"] == 12_300


def test_akshare_tencent_ambiguous_volume_unit_fails() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2026-01-05"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [1_000.0],
            "amount": [100_000.0],
            "turnover": [0.012],
        }
    )
    with pytest.raises(ValueError, match="ambiguous"):
        normalize_akshare_tx_bars(raw, "600000", datetime.now(UTC), "raw", "req")


def test_baostock_mapping_and_units() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2026-01-05"],
            "code": ["sh.600000"],
            "open": ["10"],
            "high": ["11"],
            "low": ["9"],
            "close": ["10.5"],
            "volume": ["12300"],
            "amount": ["129150"],
            "turn": ["1.2"],
            "tradestatus": ["1"],
            "isST": ["0"],
        }
    )
    result = normalize_baostock_bars(raw, "sh.600000", datetime.now(UTC), "raw", "req")
    assert result.loc[0, "volume"] == 12_300
    assert result.loc[0, "turnover"] == pytest.approx(0.012)
    assert bool(result.loc[0, "is_trading"])


def test_schema_change_is_explicit() -> None:
    with pytest.raises(ValueError, match="schema changed"):
        normalize_akshare_bars(
            pd.DataFrame({"日期": []}),
            "600000",
            datetime.now(UTC),
            "raw",
            "req",
        )


def test_baostock_cursor_conversion_avoids_removed_dataframe_append() -> None:
    class Result:
        error_code = "0"
        error_msg = ""
        fields = ["date", "code"]

        def __init__(self) -> None:
            self.rows = iter([["2026-01-05", "sh.600000"], ["2026-01-06", "sh.600000"]])
            self.current = None

        def next(self) -> bool:
            self.current = next(self.rows, None)
            return self.current is not None

        def get_row_data(self):
            return self.current

    frame = BaostockAdapter._result_frame(Result())
    assert frame.shape == (2, 2)
