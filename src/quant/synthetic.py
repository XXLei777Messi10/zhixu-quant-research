from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from quant.data.normalize import STANDARD_COLUMNS


def synthetic_normalized(
    symbols: int = 24,
    periods: int = 1600,
    start: str = "2010-01-04",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=periods)
    names = [f"SH{600000 + index:06d}" for index in range(symbols)]
    names.append("SH000300")
    frames: list[pd.DataFrame] = []
    retrieved = pd.Timestamp(datetime.now(UTC))
    for index, symbol in enumerate(names):
        market = index == len(names) - 1
        drift = 0.00015 if market else 0.0002 + (index % 5) * 0.00002
        shocks = rng.normal(drift, 0.012 if market else 0.018, len(dates))
        raw_close = (10 + index * 0.2) * np.exp(np.cumsum(shocks))
        overnight = rng.normal(0, 0.003, len(dates))
        raw_open = raw_close / (1 + rng.normal(0, 0.008, len(dates))) * (1 + overnight)
        spread = np.abs(rng.normal(0.012, 0.004, len(dates)))
        raw_high = np.maximum(raw_open, raw_close) * (1 + spread)
        raw_low = np.minimum(raw_open, raw_close) * (1 - spread)
        volume = rng.integers(1_000_000, 20_000_000, len(dates)).astype(float)
        amount = volume * raw_close
        turnover = rng.uniform(0.002, 0.04, len(dates))
        factor = np.ones(len(dates))
        if not market and index % 7 == 0 and len(dates) > 700:
            factor[700:] = 1.1
        for source, noise_scale in (("akshare", 0.0), ("baostock", 0.00005)):
            noise = 1 + rng.normal(0, noise_scale, len(dates))
            frame = pd.DataFrame(
                {
                    "symbol": symbol,
                    "trade_date": dates,
                    "open": raw_open * noise,
                    "high": raw_high * noise,
                    "low": raw_low * noise,
                    "close": raw_close * noise,
                    "volume": volume,
                    "amount": amount,
                    "turnover": turnover,
                    "adjust_factor": np.nan,
                    "is_trading": True,
                    "source": source,
                    "retrieved_at": retrieved,
                    "adjustment": "raw",
                    "quality_status": "UNCHECKED",
                    "request_id": f"synthetic-{source}-{symbol}",
                    "is_st": False,
                }
            )
            frames.append(frame[STANDARD_COLUMNS])
        hfq = pd.DataFrame(
            {
                "symbol": symbol,
                "trade_date": dates,
                "open": raw_open * factor,
                "high": raw_high * factor,
                "low": raw_low * factor,
                "close": raw_close * factor,
                "volume": volume,
                "amount": amount,
                "turnover": turnover,
                "adjust_factor": factor,
                "is_trading": True,
                "source": "akshare",
                "retrieved_at": retrieved,
                "adjustment": "hfq",
                "quality_status": "UNCHECKED",
                "request_id": f"synthetic-akshare-hfq-{symbol}",
                "is_st": False,
            }
        )
        frames.append(hfq[STANDARD_COLUMNS])
    return pd.concat(frames, ignore_index=True)
