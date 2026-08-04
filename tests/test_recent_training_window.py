from __future__ import annotations

import pandas as pd
import pytest

from quant.models.splits import TimeSplit
from quant.models.training import effective_training_start


def split() -> TimeSplit:
    return TimeSplit(
        train_start=pd.Timestamp("2010-01-04"),
        train_end=pd.Timestamp("2023-01-03"),
        valid_start=pd.Timestamp("2023-01-11"),
        valid_end=pd.Timestamp("2023-12-29"),
        test_start=pd.Timestamp("2024-01-08"),
        test_end=pd.Timestamp("2024-03-29"),
    )


def test_recent_training_window_excludes_old_rows() -> None:
    assert effective_training_start(
        split(),
        {"training_window_years": 5},
    ) == pd.Timestamp("2018-01-03")


def test_expanding_window_remains_default() -> None:
    assert effective_training_start(split(), {}) == pd.Timestamp("2010-01-04")


def test_recent_training_window_requires_three_years() -> None:
    with pytest.raises(ValueError, match="at least three"):
        effective_training_start(split(), {"training_window_years": 2})
