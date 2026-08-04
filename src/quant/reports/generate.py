from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from quant.signals.archive import immutable_write_bytes, immutable_write_json


def generate_research_report(
    metrics: dict[str, Any],
    output_dir: Path,
    name: str = "latest",
    limitations: list[str] | None = None,
) -> tuple[Path, Path]:
    limitations = limitations or [
        "免费接口可能变更或修订历史数据。",
        "即使使用历史成分，退市和临时调整数据仍可能不完整。",
        "日线模拟无法知道开盘集合竞价后的真实可成交量。",
        "历史回测和模拟结果不能证明未来盈利。",
    ]
    json_path = immutable_write_json(
        output_dir / f"{name}.json", {"metrics": metrics, "limitations": limitations}
    )
    metric_rows = "".join(
        "<tr><th>"
        + html.escape(str(key))
        + "</th><td><pre>"
        + html.escape(json.dumps(value, ensure_ascii=False, default=str))
        + "</pre></td></tr>"
        for key, value in metrics.items()
    )
    limitation_items = "".join(f"<li>{html.escape(item)}</li>" for item in limitations)
    page = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>A股量化研究报告</title>
<style>body{{font-family:sans-serif;max-width:1100px;margin:32px auto;line-height:1.5}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px;text-align:left}}
th{{background:#f4f4f4;width:28%}}.warning{{background:#fff3cd;padding:12px}}</style>
<h1>A股量化研究报告</h1>
<p class="warning">仅供个人离线模拟研究，不构成投资建议，不代表或保证盈利。</p>
<h2>指标</h2><table>{metric_rows}</table>
<h2>失败区间与已知限制</h2><ul>{limitation_items}</ul></html>"""
    html_path = immutable_write_bytes(output_dir / f"{name}.html", page.encode("utf-8"))
    return json_path, html_path
