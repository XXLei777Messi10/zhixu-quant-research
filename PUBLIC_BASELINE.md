# ZhiXu Rank Baseline

## Current public release

- Public release: v0.2.0.
- Selected configuration: configs/rank_buffer_final_accounting_repair.yaml.
- Lineage label: rank-buffer-v4-locked-data-accounting-repair.
- This public release is intentionally older than the internal research line and is not the main-account production model.

## Public scope

- Project: ZhiXu Quant Research（知序量研）.
- Model: ZhiXu Rank Buffer Baseline（知序排名缓冲公开基线）.
- Public portfolio configuration: configs/rank_buffer_final_accounting_repair.yaml.
- Simulation only; no broker connection or real order execution.

## v0.2.0 fixed portfolio and execution parameters

| Parameter | Value |
|---|---:|
| candidate pool | weekly Top20 |
| maximum weekly replacements | 2 |
| exit rank | 30 |
| execution | next trading day open |
| weighting | conviction × inverse volatility |
| score weight | 0.65 |
| probability weight | 0.35 |
| minimum conviction | 0.10 |
| volatility floor | 0.12 |
| volatility power | 0.50 |
| maximum single-stock weight | 0.10 |
| maximum sector weight | 0.25 |
| no-trade weight band | 0.02 |
| no-trade cost multiple | 8.0 |
| open filter | walk-forward gap calibration |
| open-filter mode | mild cost-aware |
| lot size | 100 |

The public configuration also records a historical membership data repair and an accounting repair for adjusted-unit share residuals. These repairs are part of reproducibility and accounting integrity; they are not performance claims.

## Execution boundaries

- Signals are generated after the close and are simulated no earlier than the next trading-day open.
- The simulation includes commission, stamp duty, transfer fee, slippage, suspensions, price limits, T+1 and 100-share lot size.
- Orders that violate the public execution rules must not be silently treated as filled.
- External free data may be rate-limited, revised or unavailable. Re-running on a different snapshot may produce different research outputs.

The public package intentionally excludes deployment material, private runtime data, account history, trained artifacts, non-public research lineages, and all recorded model-performance results.

## No performance claims

This repository intentionally publishes no measured return, excess return, prediction accuracy, IC, drawdown, Sharpe ratio, information ratio, or other model-capability result. It makes no claim that the model is profitable, superior to a benchmark, suitable for investment, or likely to achieve any particular outcome.

The v0.2.0 release is for quantitative research, software engineering demonstration and academic exchange only. It is not investment advice, a securities recommendation, a promise of returns, or a basis for trading. The public model is different from the main-account production model.
