"""Compare transaction-cost assumptions on fixed synthetic research data."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostScenario:
    name: str
    cost_bps: float


GROSS_PNL_BPS = (42.0, -18.0, 36.0, -12.0, 28.0)
TURNOVER = (0.8, 0.6, 1.0, 0.4, 0.8)
SCENARIOS = (
    CostScenario("zero_cost", 0.0),
    CostScenario("base_cost", 8.0),
    CostScenario("stress_cost", 16.0),
)


def calculate(scenario: CostScenario) -> tuple[float, float, float]:
    gross_pnl_bps = sum(GROSS_PNL_BPS)
    total_turnover = sum(TURNOVER)
    total_cost_bps = total_turnover * scenario.cost_bps
    net_pnl_bps = gross_pnl_bps - total_cost_bps
    return gross_pnl_bps, total_cost_bps, net_pnl_bps


def main() -> None:
    print("ZhiXu transaction-cost sensitivity demo")
    print("data_source: deterministic synthetic")
    print("status: ok")
    print("scenario       gross_pnl_bps  total_cost_bps  net_pnl_bps")
    for scenario in SCENARIOS:
        gross, total_cost, net = calculate(scenario)
        print(f"{scenario.name:<12} {gross:>14.1f} {total_cost:>15.1f} {net:>13.1f}")
    print("live_data: false")


if __name__ == "__main__":
    main()
