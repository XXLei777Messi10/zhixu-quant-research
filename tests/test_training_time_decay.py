from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.models.training import training_sample_weights


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2015-01-02", "2020-01-02", "2025-01-02"]
            )
        }
    )


def config(half_life_years: float = 5) -> dict[str, object]:
    return {
        "training_sample_weight": {
            "method": "exponential_time_decay",
            "half_life_years": half_life_years,
            "normalization": "mean_one",
        }
    }


def test_time_decay_retains_all_rows_and_normalizes_mean() -> None:
    weights = training_sample_weights(frame(), config())
    assert weights is not None
    assert len(weights) == len(frame())
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-6)
    assert weights.iloc[0] < weights.iloc[1] < weights.iloc[2]


def test_time_decay_has_configured_half_life() -> None:
    weights = training_sample_weights(frame(), config())
    assert weights is not None
    assert float(weights.iloc[1] / weights.iloc[2]) == pytest.approx(
        0.5, abs=2e-4
    )


def test_training_weights_remain_disabled_by_default() -> None:
    assert training_sample_weights(frame(), {}) is None


@pytest.mark.parametrize("half_life", [0, -1, np.nan])
def test_time_decay_rejects_invalid_half_life(half_life: float) -> None:
    with pytest.raises(ValueError, match="half-life"):
        training_sample_weights(frame(), config(half_life))


def test_time_decay_rejects_unknown_method() -> None:
    value = config()
    value["training_sample_weight"]["method"] = "linear"
    with pytest.raises(ValueError, match="Unsupported"):
        training_sample_weights(frame(), value)
