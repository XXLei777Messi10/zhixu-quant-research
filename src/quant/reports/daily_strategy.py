from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from quant.config import ProjectPaths
from quant.execution.archive import load_execution_plans
from quant.execution.models import ExecutionPlan
from quant.signals.archive import immutable_write_bytes, immutable_write_json


def _state(paths: ProjectPaths, variant: str) -> dict[str, Any]:
    path = paths.state / "simulation" / variant / "current.json"
    if not path.exists():
        return {
            "variant": variant,
            "status": "NOT_INITIALIZED",
            "cash": None,
            "nav": None,
            "positions": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "variant": variant,
        "status": "AVAILABLE",
        "as_of": payload.get("as_of"),
        "cash": payload.get("cash"),
        "market_value": payload.get("market_value"),
        "nav": payload.get("nav"),
        "gross_exposure": (
            float(payload.get("market_value", 0.0)) / float(payload["nav"])
            if payload.get("nav")
            else 0.0
        ),
        "positions": payload.get("positions", []),
    }


def _plan_summary(plan: ExecutionPlan) -> dict[str, Any]:
    return {
        "rank": plan.signal_rank,
        "symbol": plan.symbol,
        "name": plan.name,
        "model_signal": plan.model_signal,
        "initial_state": plan.initial_state,
        "current_weight": plan.current_position_weight,
        "target_weight": plan.target_position_weight,
        "initial_order_weight": plan.initial_order_weight,
        "add_order_weight": plan.add_order_weight,
        "buy_range": [plan.entry_price_low, plan.entry_price_high],
        "reduced_entry_until": plan.chase_limit_price,
        "invalidation_below": plan.invalidation_price,
        "hard_exit_below": plan.hard_exit_price,
        "reduce_from": plan.reduce_price,
        "market_state": plan.market_state,
        "sector_name": plan.sector_name,
        "sector_state": plan.sector_state,
        "model_score": plan.model_score,
        "outperform_probability": plan.outperform_probability,
        "predicted_excess_return_5d": plan.predicted_excess_return_5d,
        "parameter_source": plan.parameter_source,
    }


def _account_rows(accounts: dict[str, dict[str, Any]]) -> str:
    rows = []
    for role, account in accounts.items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(role)}</td>"
            f"<td>{html.escape(str(account['variant']))}</td>"
            f"<td>{html.escape(str(account.get('cash')))}</td>"
            f"<td>{html.escape(str(account.get('nav')))}</td>"
            f"<td>{html.escape(str(account.get('gross_exposure')))}</td>"
            f"<td>{len(account.get('positions', []))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _plan_rows(plans: list[dict[str, Any]], role: str) -> str:
    rows = []
    for plan in plans:
        rows.append(
            "<tr>"
            f"<td>{html.escape(role)}</td>"
            f"<td>{plan['rank']}</td>"
            f"<td>{html.escape(plan['symbol'])}</td>"
            f"<td>{html.escape(plan['name'])}</td>"
            f"<td>{html.escape(plan['model_signal'])}</td>"
            f"<td>{html.escape(plan['initial_state'])}</td>"
            f"<td>{plan['current_weight']:.2%}</td>"
            f"<td>{plan['target_weight']:.2%}</td>"
            f"<td>{plan['buy_range'][0]}–{plan['buy_range'][1]}</td>"
            f"<td>{plan['reduced_entry_until']}</td>"
            f"<td>{plan['invalidation_below']}</td>"
            f"<td>{html.escape(str(plan['sector_state']))}</td>"
            "</tr>"
        )
    return "".join(rows)


def generate_daily_strategy_summary(
    paths: ProjectPaths,
    signals: pd.DataFrame,
    signal_date: str,
    execution_date: str,
    gated_plan_path: Path,
    ungated_plan_path: Path,
    candidate_pool: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, str]:
    gated_plans = [
        _plan_summary(plan) for plan in load_execution_plans(gated_plan_path)
    ]
    ungated_plans = [
        _plan_summary(plan) for plan in load_execution_plans(ungated_plan_path)
    ]
    accounts = {
        "primary_model_account": _state(paths, "gated"),
        "shadow_model_account": _state(paths, "ungated"),
        "user_manual_comparison": _state(paths, "manual"),
    }
    manual_symbols = {
        str(item["symbol"])
        for item in accounts["user_manual_comparison"].get("positions", [])
    }
    manual_review = (
        signals[signals["symbol"].astype(str).isin(manual_symbols)]
        .sort_values(["signal_rank", "symbol"])
        .to_dict("records")
    )
    payload = {
        "signal_date": signal_date,
        "execution_date": execution_date,
        "account_roles": {
            "gated": "PRIMARY_MODEL_ACCOUNT_AUTOMATIC",
            "ungated": "SHADOW_MODEL_ACCOUNT_AUTOMATIC",
            "manual": "USER_REPORTED_COMPARISON_NO_AUTOMATIC_ORDERS",
        },
        "authoritative_daily_policy": policy,
        "candidate_pool": candidate_pool,
        "accounts_before_execution": accounts,
        "gated_primary_plans": gated_plans,
        "ungated_shadow_plans": ungated_plans,
        "manual_position_model_review": manual_review,
        "reporting_rule": (
            "Daily summaries must preserve these account roles and must not treat "
            "user-reported manual fills as model fills."
        ),
        "simulation_only": True,
    }
    output_dir = paths.reports / "daily-strategy"
    json_path = immutable_write_json(
        output_dir / f"{signal_date}.json",
        payload,
    )
    page = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>{signal_date} 次日策略汇总</title>
<style>
body{{font-family:Arial,"Microsoft YaHei",sans-serif;max-width:1500px;margin:24px auto;line-height:1.5}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:24px}}
th,td{{border:1px solid #ddd;padding:6px;text-align:left}}th{{background:#f4f4f4}}
.warning{{background:#fff3cd;padding:10px}}code{{background:#f5f5f5;padding:2px 4px}}
</style>
<h1>{signal_date} 收盘后 / {execution_date} 执行策略</h1>
<p class="warning">仅供个人模拟研究。主账户=gated；影子账户=ungated；manual仅记录用户报告成交。</p>
<h2>统一生产规则</h2><pre>{html.escape(json.dumps(policy, ensure_ascii=False, indent=2))}</pre>
<h2>三个账户</h2>
<table><tr><th>角色</th><th>账户</th><th>现金</th><th>净值</th><th>股票仓位</th><th>持股数</th></tr>
{_account_rows(accounts)}</table>
<h2>主账户与影子账户次日计划</h2>
<table><tr><th>账户</th><th>排名</th><th>代码</th><th>名称</th><th>信号</th><th>状态</th>
<th>当前</th><th>目标</th><th>正常买入区间</th><th>降级买入上限</th><th>失效线</th><th>行业状态</th></tr>
{_plan_rows(gated_plans, "gated主账户")}
{_plan_rows(ungated_plans, "ungated影子账户")}</table>
<h2>人工对照账户持仓的模型核对</h2>
<pre>{html.escape(json.dumps(manual_review, ensure_ascii=False, indent=2, default=str))}</pre>
</html>"""
    html_path = immutable_write_bytes(
        output_dir / f"{signal_date}.html",
        page.encode("utf-8"),
    )
    return {"json": str(json_path), "html": str(html_path)}
