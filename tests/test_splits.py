import pandas as pd
import pytest

from quant.models.splits import assert_no_overlap, current_training_split, quarterly_splits


def test_time_splits_have_double_embargo() -> None:
    calendar = pd.bdate_range("2010-01-01", "2015-12-31")
    splits = quarterly_splits(calendar, "2014-01-01", 12, 5)
    assert splits
    for split in splits:
        assert_no_overlap(split, 5, calendar)
        assert split.train_start < split.train_end < split.valid_start < split.valid_end < split.test_start


def test_current_split_is_chronological() -> None:
    calendar = pd.bdate_range("2010-01-01", "2015-12-31")
    split = current_training_split(calendar)
    assert split.train_end < split.valid_start
    assert split.valid_end < split.test_start


def test_short_training_history_is_rejected() -> None:
    calendar = pd.bdate_range("2023-01-01", "2026-01-01")
    with pytest.raises(ValueError, match="shorter than 3 years"):
        current_training_split(calendar)
