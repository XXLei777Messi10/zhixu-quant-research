from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def normalized(self) -> Path:
        return self.root / "data" / "normalized"

    @property
    def curated(self) -> Path:
        return self.root / "data" / "curated"

    @property
    def quarantine(self) -> Path:
        return self.root / "data" / "quarantine"

    @property
    def qlib(self) -> Path:
        return self.root / "data" / "qlib"

    @property
    def models(self) -> Path:
        return self.root / "artifacts" / "models"

    @property
    def state(self) -> Path:
        return self.root / "artifacts" / "state"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def ensure_runtime_dirs(self) -> None:
        for path in (
            self.raw,
            self.normalized,
            self.curated,
            self.quarantine,
            self.qlib,
            self.models,
            self.state,
            self.reports / "signals",
            self.reports / "data-quality",
            self.reports / "portfolio",
            self.reports / "research",
            self.reports / "execution-plans",
            self.reports / "shadow-execution-plans",
            self.reports / "simulation" / "gated" / "accounts",
            self.reports / "simulation" / "gated" / "orders",
            self.reports / "simulation" / "gated" / "fills",
            self.reports / "simulation" / "ungated" / "accounts",
            self.reports / "simulation" / "ungated" / "orders",
            self.reports / "simulation" / "ungated" / "fills",
            self.reports / "simulation" / "manual" / "accounts",
            self.reports / "simulation" / "manual" / "trades",
            self.reports / "daily-strategy",
            self.reports / "operations" / "daily-verification",
            self.reports / "auction",
            self.reports / "orders",
            self.reports / "fills",
            self.reports / "positions",
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return value or {}


def load_config(paths: ProjectPaths, name: str) -> dict[str, Any]:
    return load_yaml(paths.configs / f"{name}.yaml")
