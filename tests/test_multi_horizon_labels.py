from __future__ import annotations

import numpy as np
import pandas as pd

from quant.research.model_components import build_forward_labels


def _bars() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=30)
    benchmark = pd.DataFrame(
        {
            "symbol": "SH000300",
            "trade_date": dates,
            "hfq_open": 100.0 * np.power(1.001, np.arange(len(dates))),
        }
    )
    stock = pd.DataFrame(
        {
            "symbol": "SH600000",
            "trade_date": dates,
            "hfq_open": 10.0 * np.power(1.003, np.arange(len(dates))),
        }
    )
    return pd.concat([benchmark, stock], ignore_index=True)


def test_forward_labels_support_5_10_and_20_day_horizons() -> None:
    bars = _bars()
    for horizon in (5, 10, 20):
        labels = build_forward_labels(bars, "SH000300", horizon=horizon)
        expected = (1.003**horizon - 1.0) - (1.001**horizon - 1.0)
        assert np.isclose(labels.iloc[0]["label_regression"], expected)
        mature = labels["label_regression"].notna().sum()
        assert mature == len(labels) - horizon - 1


def test_forward_label_never_uses_t_close_as_entry() -> None:
    bars = _bars()
    labels = build_forward_labels(bars, "SH000300", horizon=10, entry_offset=1)
    changed_bars = bars.copy()
    first_stock = changed_bars.index[changed_bars["symbol"].eq("SH600000")][0]
    changed_bars.loc[first_stock, "hfq_open"] = 9999.0
    changed = build_forward_labels(changed_bars, "SH000300", horizon=10)
    # The T-day opening price is not part of the T+1-to-T+11 label.
    assert labels.iloc[0]["label_regression"] == changed.iloc[0]["label_regression"]


def test_invalid_horizon_is_rejected() -> None:
    try:
        build_forward_labels(_bars(), "SH000300", horizon=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("Zero horizon should have been rejected")
