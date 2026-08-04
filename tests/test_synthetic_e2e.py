from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml


@pytest.mark.slow
def test_all_eight_primary_cli_commands_on_synthetic_data(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    shutil.copytree(project / "configs", tmp_path / "configs")
    model_path = tmp_path / "configs" / "model.yaml"
    model = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    model["lightgbm"]["num_boost_round"] = 80
    model["lightgbm"]["early_stopping_rounds"] = 10
    model_path.write_text(
        yaml.safe_dump(model, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    commands = [
        ["fetch", "--synthetic"],
        ["validate-data"],
        ["build-dataset"],
        ["train"],
        ["backtest", "--max-folds", "1"],
        ["predict"],
        ["report"],
        ["run-daily", "--synthetic"],
    ]
    for command in commands:
        completed = subprocess.run(
            [sys.executable, "-m", "quant", "--root", str(tmp_path), *command],
            cwd=project,
            text=True,
            capture_output=True,
            timeout=600,
            check=False,
        )
        assert completed.returncode == 0, (
            f"{' '.join(command)} failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
        payload_start = completed.stdout.rfind("\n{")
        payload_text = completed.stdout[payload_start + 1 :] if payload_start >= 0 else completed.stdout
        assert isinstance(json.loads(payload_text), dict)

    signal_files = list((tmp_path / "reports" / "signals").glob("*.csv"))
    assert signal_files
    signal_frame = pd.read_csv(sorted(signal_files)[-1], encoding="utf-8-sig")
    assert {
        "buy_price_low",
        "buy_price_high",
        "sell_price_low",
        "sell_price_high",
        "price_range_status",
    }.issubset(signal_frame.columns)
    assert signal_frame["buy_price_low"].lt(signal_frame["buy_price_high"]).all()
    assert signal_frame["sell_price_low"].le(signal_frame["sell_price_high"]).all()
    assert signal_frame["sector_state"].ne("DATA_UNAVAILABLE").all()
    assert signal_frame["sector_name"].notna().all()
    assert list((tmp_path / "reports" / "execution-plans").glob("*.json"))
    assert list((tmp_path / "reports" / "shadow-execution-plans").glob("*.json"))
    assert list(
        (tmp_path / "reports" / "simulation" / "gated" / "accounts").glob("*.json")
    )
    assert list(
        (tmp_path / "reports" / "simulation" / "ungated" / "accounts").glob("*.json")
    )
    assert (tmp_path / "artifacts" / "models" / "current.json").exists()
