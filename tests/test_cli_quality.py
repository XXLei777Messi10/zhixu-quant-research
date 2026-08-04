import json
from pathlib import Path

from quant.cli import _latest_quality_status, generate_report_command
from quant.config import ProjectPaths


def test_latest_quality_revision_is_authoritative(tmp_path) -> None:
    paths = ProjectPaths(tmp_path)
    directory = paths.reports / "data-quality"
    directory.mkdir(parents=True)
    (directory / "2026-01-05.json").write_text(
        json.dumps({"status": "FAIL"}),
        encoding="utf-8",
    )
    (directory / "2026-01-05__r2.json").write_text(
        json.dumps({"status": "PASS"}),
        encoding="utf-8",
    )
    assert _latest_quality_status(paths, "2026-01-05") == "PASS"


def test_report_command_uses_latest_immutable_backtest_revision(tmp_path) -> None:
    paths = ProjectPaths(tmp_path)
    directory = paths.reports / "research"
    directory.mkdir(parents=True)
    (directory / "backtest.json").write_text(
        json.dumps({"metrics": {"annualized_return": -1.0}}),
        encoding="utf-8",
    )
    (directory / "backtest__r2.json").write_text(
        json.dumps({"metrics": {"annualized_return": 0.1}}),
        encoding="utf-8",
    )
    (directory / "backtest__r10.json").write_text(
        json.dumps({"metrics": {"annualized_return": 0.2}}),
        encoding="utf-8",
    )

    result = generate_report_command(paths)
    payload = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
    assert payload["metrics"]["annualized_return"] == 0.2
