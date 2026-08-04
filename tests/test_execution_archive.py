import json

from quant.execution.archive import resolve_execution_plan_path


def test_execution_plan_loader_resolves_numeric_latest_revision(tmp_path) -> None:
    for name in ("2026-01-06.json", "2026-01-06__r2.json", "2026-01-06__r10.json"):
        (tmp_path / name).write_text(json.dumps({"plans": []}), encoding="utf-8")

    selected = resolve_execution_plan_path(tmp_path / "2026-01-06.json")
    assert selected.name == "2026-01-06__r10.json"
