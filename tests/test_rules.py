from datetime import date

import pytest

from quant.backtest.rules import (
    is_order_blocked_by_limit,
    price_limit_ratio,
    rounded_limit_price,
    stamp_duty_rate,
)


def test_board_limit_rules() -> None:
    assert price_limit_ratio("SH600000", date(2026, 1, 1)) == 0.10
    assert price_limit_ratio("SH688001", date(2026, 1, 1)) == 0.20
    assert price_limit_ratio("SZ300001", date(2020, 8, 21)) == 0.10
    assert price_limit_ratio("SZ300001", date(2020, 8, 24)) == 0.20
    assert price_limit_ratio("BJ830001", date(2026, 1, 1)) == 0.30
    assert price_limit_ratio("SH600000", date(2026, 1, 1), True) == 0.05


def test_limit_rounding_and_blocking() -> None:
    assert rounded_limit_price(10.03, 0.10, "up") == 11.03
    assert is_order_blocked_by_limit("BUY", 11.03, 10.03, "SH600000", date(2026, 1, 1))
    assert not is_order_blocked_by_limit("BUY", 11.02, 10.03, "SH600000", date(2026, 1, 1))


def test_stamp_schedule() -> None:
    schedule = [
        {"start": "1900-01-01", "end": "2023-08-27", "rate": 0.001},
        {"start": "2023-08-28", "end": "2099-12-31", "rate": 0.0005},
    ]
    assert stamp_duty_rate(date(2023, 8, 27), schedule) == pytest.approx(0.001)
    assert stamp_duty_rate(date(2023, 8, 28), schedule) == pytest.approx(0.0005)
