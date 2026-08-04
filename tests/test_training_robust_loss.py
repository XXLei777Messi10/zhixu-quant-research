from __future__ import annotations

import pytest

from quant.models.training import lightgbm_regression_parameters


def test_regression_objective_remains_l2_by_default() -> None:
    params = lightgbm_regression_parameters({"learning_rate": 0.03}, {})

    assert params["objective"] == "regression"
    assert params["metric"] == "l2"
    assert "alpha" not in params


def test_huber_objective_uses_pre_registered_alpha() -> None:
    params = lightgbm_regression_parameters(
        {"learning_rate": 0.03},
        {"lightgbm_regression_objective": {"name": "huber", "alpha": 0.90}},
    )

    assert params["objective"] == "huber"
    assert params["metric"] == "l2"
    assert params["alpha"] == pytest.approx(0.90)


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1])
def test_huber_objective_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="Huber alpha"):
        lightgbm_regression_parameters(
            {},
            {"lightgbm_regression_objective": {"name": "huber", "alpha": alpha}},
        )


def test_regression_objective_rejects_unknown_loss() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        lightgbm_regression_parameters(
            {},
            {"lightgbm_regression_objective": {"name": "quantile"}},
        )
