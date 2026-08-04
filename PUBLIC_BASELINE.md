# ZhiXu Rank Baseline

## Public scope

- Project: ZhiXu Quant Research（知序量研）.
- Model: ZhiXu Rank Baseline（知序横截面排序基线）.
- Public portfolio configuration: `configs/portfolio_baseline_final.yaml`.
- Simulation only; no broker connection or real order execution.

The public package intentionally excludes deployment material, private runtime data, account history, trained artifacts, non-public research lineages, and all recorded model-performance results.

## Fixed portfolio parameters

| Parameter | Value |
|---|---:|
| model variant | `zhixu_rank_baseline` |
| target holdings | 20 |
| maximum weekly replacements | 2 |
| exit rank | 30 |
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
| open filter | disabled |
| market-state exposure scaling | disabled |

## No performance claims

This repository intentionally publishes no measured return, excess return, prediction accuracy, IC, drawdown, Sharpe ratio, information ratio, or other model-capability result. It makes no claim that the model is profitable, superior to a benchmark, suitable for investment, or likely to achieve any particular outcome.
