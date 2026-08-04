from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import pandas as pd

from quant.data.normalize import (
    canonical_symbol,
    infer_akshare_tx_volume_multiplier,
    normalize_akshare_bars,
    normalize_akshare_tx_bars,
    provider_symbol,
)
from quant.data.raw_archive import RawArchive


@dataclass
class AKShareAdapter:
    archive: RawArchive
    timeout_seconds: int = 20
    requests_per_second: float = 2.0
    _volume_multipliers: dict[str, float] = field(default_factory=dict, init=False)

    def _module(self):
        import akshare

        return akshare

    def fetch_stock_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        adjustment: str = "raw",
    ) -> pd.DataFrame:
        ak = self._module()
        adjust_arg = "" if adjustment == "raw" else adjustment
        params = {
            "symbol": provider_symbol(symbol, "akshare"),
            "period": "daily",
            "start_date": start.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
            "adjust": adjust_arg,
            "timeout": self.timeout_seconds,
        }
        try:
            raw = ak.stock_zh_a_hist(**params)
            if raw.empty:
                raise RuntimeError("AKShare Eastmoney endpoint returned no rows")
        except Exception as primary_error:
            self.archive.save(
                "akshare",
                "stock_zh_a_hist_error",
                pd.DataFrame(
                    {
                        "error_type": [type(primary_error).__name__],
                        "message": [str(primary_error)],
                    }
                ),
                params,
                ak.__version__,
                datetime.now(UTC),
            )
            return self._fetch_tencent_bars(symbol, start, end, adjustment)
        retrieved_at = datetime.now(UTC)
        snapshot = self.archive.save("akshare", "stock_zh_a_hist", raw, params, ak.__version__, retrieved_at)
        if self.requests_per_second > 0:
            time.sleep(1.0 / self.requests_per_second)
        return normalize_akshare_bars(
            raw,
            canonical_symbol(symbol),
            retrieved_at,
            adjustment,
            snapshot.request_id,
            volume_in_lots=True,
        )

    def _fetch_tencent_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        adjustment: str,
    ) -> pd.DataFrame:
        canonical = canonical_symbol(symbol)
        if canonical.startswith("BJ"):
            raise RuntimeError("AKShare Tencent fallback does not support Beijing symbols")
        ak = self._module()
        params = {
            "symbol": provider_symbol(canonical, "akshare_tx"),
            "start_date": start.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
            "adjust": "" if adjustment == "raw" else adjustment,
            "timeout": self.timeout_seconds,
        }
        raw = ak.stock_zh_a_hist_tx(**params)
        if raw.empty:
            raise RuntimeError("Both AKShare daily endpoints returned no rows")
        if adjustment == "raw":
            multiplier = infer_akshare_tx_volume_multiplier(raw)
            self._volume_multipliers[canonical] = multiplier
        else:
            multiplier = self._volume_multipliers.get(canonical)
            if multiplier is None:
                raw_unit_sample = ak.stock_zh_a_hist_tx(**{**params, "adjust": ""})
                multiplier = infer_akshare_tx_volume_multiplier(raw_unit_sample)
                self._volume_multipliers[canonical] = multiplier
        retrieved_at = datetime.now(UTC)
        snapshot = self.archive.save(
            "akshare",
            "stock_zh_a_hist_tx",
            raw,
            params,
            ak.__version__,
            retrieved_at,
        )
        if self.requests_per_second > 0:
            time.sleep(1.0 / self.requests_per_second)
        return normalize_akshare_tx_bars(
            raw,
            canonical,
            retrieved_at,
            adjustment,
            snapshot.request_id,
            multiplier,
        )

    def fetch_index_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        ak = self._module()
        code = canonical_symbol(symbol)[2:]
        params = {
            "symbol": code,
            "period": "daily",
            "start_date": start.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
        }
        try:
            raw = ak.index_zh_a_hist(**params)
            if raw.empty:
                raise RuntimeError("AKShare index endpoint returned no rows")
        except Exception as primary_error:
            self.archive.save(
                "akshare",
                "index_zh_a_hist_error",
                pd.DataFrame(
                    {
                        "error_type": [type(primary_error).__name__],
                        "message": [str(primary_error)],
                    }
                ),
                params,
                ak.__version__,
                datetime.now(UTC),
            )
            return self._fetch_tencent_bars(symbol, start, end, "hfq")
        retrieved_at = datetime.now(UTC)
        snapshot = self.archive.save("akshare", "index_zh_a_hist", raw, params, ak.__version__, retrieved_at)
        return normalize_akshare_bars(
            raw,
            canonical_symbol(symbol),
            retrieved_at,
            "hfq",
            snapshot.request_id,
            volume_in_lots=True,
        )

    def fetch_current_csi300(self) -> pd.DataFrame:
        ak = self._module()
        params = {"symbol": "000300"}
        raw = ak.index_stock_cons_csindex(**params)
        self.archive.save(
            "akshare",
            "index_stock_cons_csindex",
            raw,
            params,
            ak.__version__,
            datetime.now(UTC),
        )
        return raw
