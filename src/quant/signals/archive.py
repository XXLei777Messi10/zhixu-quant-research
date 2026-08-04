from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import pandas as pd


def _revision_path(path: Path) -> Path:
    revision = 2
    while True:
        candidate = path.with_name(f"{path.stem}__r{revision}{path.suffix}")
        if not candidate.exists():
            return candidate
        revision += 1


def immutable_write_bytes(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    content_hash = hashlib.sha256(content).digest()
    candidates = [path, *path.parent.glob(f"{path.stem}__r*{path.suffix}")]
    matching = [
        candidate
        for candidate in candidates
        if candidate.exists()
        and hashlib.sha256(candidate.read_bytes()).digest() == content_hash
    ]
    if matching:
        def revision(candidate: Path) -> int:
            if candidate == path:
                return 1
            try:
                return int(candidate.stem.rsplit("__r", maxsplit=1)[1])
            except (IndexError, ValueError):
                return 0

        return max(matching, key=revision)
    if path.exists():
        path = _revision_path(path)
    with path.open("xb") as handle:
        handle.write(content)
    return path


def archive_signals(
    frame: pd.DataFrame,
    output_dir: Path,
    trade_date: str,
) -> tuple[Path, Path]:
    required = {"model_score", "symbol"}
    if missing := required - set(frame):
        raise ValueError(f"Signal archive missing: {sorted(missing)}")
    ordered = frame.sort_values(["model_score", "symbol"], ascending=[False, True]).copy()
    csv_bytes = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
    csv_path = immutable_write_bytes(output_dir / f"{trade_date}.csv", csv_bytes)
    rows = []
    for record in ordered.to_dict("records"):
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in record.values())
        rows.append(f"<tr>{cells}</tr>")
    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in ordered.columns)
    document = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8">
<title>A股模拟候选 {trade_date}</title>
<style>body{{font-family:sans-serif;margin:24px}}table{{border-collapse:collapse;font-size:13px}}
th,td{{border:1px solid #ddd;padding:6px}}th{{background:#f4f4f4;position:sticky;top:0}}
.warning{{color:#9a5b00}}</style>
<h1>A股模拟候选 {trade_date}</h1>
<p class="warning">仅供个人研究；不是投资建议，不连接券商，不执行真实交易。</p>
<table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></html>
"""
    html_path = immutable_write_bytes(output_dir / f"{trade_date}.html", document.encode("utf-8"))
    return csv_path, html_path


def immutable_write_json(path: Path, payload: dict) -> Path:
    content = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return immutable_write_bytes(path, content)
