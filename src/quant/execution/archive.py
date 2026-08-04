from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from quant.execution.models import ExecutionPlan
from quant.signals.archive import immutable_write_bytes, immutable_write_json


def archive_execution_plans(
    plans: Iterable[ExecutionPlan],
    reports_root: Path,
    execution_date: str,
    directory: str = "execution-plans",
) -> tuple[Path, Path]:
    records = [asdict(plan) for plan in plans]
    payload = {
        "execution_date": execution_date,
        "simulation_only": True,
        "plans": records,
    }
    json_path = immutable_write_json(reports_root / directory / f"{execution_date}.json", payload)
    frame = pd.DataFrame(records)
    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
    csv_path = immutable_write_bytes(reports_root / directory / f"{execution_date}.csv", csv_bytes)
    return json_path, csv_path


def archive_execution_day(
    reports_root: Path,
    execution_date: str,
    *,
    auction: dict[str, Any],
    orders: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    positions: dict[str, Any],
) -> dict[str, str]:
    outputs = {
        "auction": immutable_write_json(reports_root / "auction" / f"{execution_date}.json", auction),
        "orders": immutable_write_json(
            reports_root / "orders" / f"{execution_date}.json",
            {"execution_date": execution_date, "simulation_only": True, "orders": orders},
        ),
        "fills": immutable_write_json(
            reports_root / "fills" / f"{execution_date}.json",
            {"execution_date": execution_date, "simulation_only": True, "fills": fills},
        ),
        "positions": immutable_write_json(reports_root / "positions" / f"{execution_date}.json", positions),
    }
    return {key: str(value) for key, value in outputs.items()}


def resolve_execution_plan_path(path: Path) -> Path:
    if "__r" in path.stem:
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    marker = f"{path.stem}__r"

    def revision(candidate: Path) -> int:
        if candidate.stem == path.stem:
            return 1
        if candidate.stem.startswith(marker):
            try:
                return int(candidate.stem[len(marker) :])
            except ValueError:
                return 0
        return 0

    candidates = list(path.parent.glob(f"{path.stem}*{path.suffix}"))
    if not candidates:
        raise FileNotFoundError(path)
    return max(candidates, key=revision)


def load_execution_plans(path: Path) -> list[ExecutionPlan]:
    path = resolve_execution_plan_path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ExecutionPlan(**item) for item in payload["plans"]]
