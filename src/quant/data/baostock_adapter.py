from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from quant.data.normalize import canonical_symbol, normalize_baostock_bars, provider_symbol
from quant.data.raw_archive import RawArchive

FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
INDUSTRY_REQUIRED = {
    "updateDate",
    "code",
    "code_name",
    "industry",
    "industryClassification",
}


class BaostockError(RuntimeError):
    pass


@dataclass
class BaostockAdapter:
    archive: RawArchive
    _active_module: Any | None = field(default=None, init=False, repr=False)
    _session_depth: int = field(default=0, init=False, repr=False)

    def _module(self):
        import baostock

        return baostock

    @staticmethod
    def _login(bs: Any) -> None:
        login = bs.login()
        if login.error_code != "0":
            raise BaostockError(f"Baostock login failed: {login.error_msg}")

    def _refresh_active_session(self) -> None:
        if self._active_module is not None:
            try:
                self._active_module.logout()
            except Exception:
                pass
        bs = self._module()
        self._login(bs)
        self._active_module = bs

    @contextmanager
    def session(self) -> Iterator[BaostockAdapter]:
        if self._active_module is not None:
            self._session_depth += 1
            try:
                yield self
            finally:
                self._session_depth -= 1
            return

        bs = self._module()
        self._login(bs)
        self._active_module = bs
        self._session_depth = 1
        try:
            yield self
        finally:
            active = self._active_module
            self._session_depth = 0
            self._active_module = None
            if active is not None:
                active.logout()

    def _with_session(self, function, *args, **kwargs):
        if self._active_module is not None:
            try:
                return function(self._active_module, *args, **kwargs)
            except BaostockError:
                self._refresh_active_session()
                raise
        with self.session():
            return function(self._active_module, *args, **kwargs)

    @staticmethod
    def _result_frame(result: Any) -> pd.DataFrame:
        rows: list[list[str]] = []
        while result.error_code == "0" and result.next():
            rows.append(result.get_row_data())
        if result.error_code != "0":
            raise BaostockError(result.error_msg)
        return pd.DataFrame(rows, columns=result.fields)

    def fetch_stock_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        adjustment: str = "raw",
    ) -> pd.DataFrame:
        adjust_map = {"hfq": "1", "qfq": "2", "raw": "3"}

        def query(bs: Any) -> pd.DataFrame:
            params = {
                "code": provider_symbol(symbol, "baostock"),
                "fields": FIELDS,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "frequency": "d",
                "adjustflag": adjust_map[adjustment],
            }
            result = bs.query_history_k_data_plus(**params)
            if result.error_code != "0":
                raise BaostockError(result.error_msg)
            raw = self._result_frame(result)
            retrieved_at = datetime.now(UTC)
            snapshot = self.archive.save(
                "baostock",
                "query_history_k_data_plus",
                raw,
                params,
                getattr(bs, "__version__", "unknown"),
                retrieved_at,
            )
            return normalize_baostock_bars(
                raw,
                canonical_symbol(symbol),
                retrieved_at,
                adjustment,
                snapshot.request_id,
            )

        return self._with_session(query)

    def fetch_historical_csi300(self, trade_date: date) -> pd.DataFrame:
        def query(bs: Any) -> pd.DataFrame:
            params = {"date": trade_date.isoformat()}
            result = bs.query_hs300_stocks(trade_date.isoformat())
            if result.error_code != "0":
                raise BaostockError(result.error_msg)
            raw = self._result_frame(result)
            self.archive.save(
                "baostock",
                "query_hs300_stocks",
                raw,
                params,
                getattr(bs, "__version__", "unknown"),
                datetime.now(UTC),
            )
            if not raw.empty:
                raw["symbol"] = raw["code"].map(canonical_symbol)
                raw["trade_date"] = pd.Timestamp(trade_date)
            return raw

        return self._with_session(query)

    def fetch_stock_industry(self, as_of: date) -> pd.DataFrame:
        def query(bs: Any) -> pd.DataFrame:
            params = {"date": as_of.isoformat()}
            result = bs.query_stock_industry(date=as_of.isoformat())
            if result.error_code != "0":
                raise BaostockError(result.error_msg)
            raw = self._result_frame(result)
            retrieved_at = datetime.now(UTC)
            snapshot = self.archive.save(
                "baostock",
                "query_stock_industry",
                raw,
                params,
                getattr(bs, "__version__", "unknown"),
                retrieved_at,
            )
            if missing := INDUSTRY_REQUIRED - set(raw):
                raise BaostockError(f"Industry contract changed, missing: {sorted(missing)}")
            if raw.empty:
                raise BaostockError(f"Industry query returned no rows for {as_of}")
            output = raw[list(INDUSTRY_REQUIRED)].copy()
            output = output[
                output["code"].astype(str).str.match(r"^(sh|sz|bj)\.\d{6}$", case=False)
            ]
            output["symbol"] = output["code"].map(canonical_symbol)
            output["stock_name"] = output["code_name"].astype(str).str.strip()
            output["industry_name"] = output["industry"].astype("string").str.strip()
            output.loc[output["industry_name"].eq(""), "industry_name"] = pd.NA
            output["industry_classification"] = (
                output["industryClassification"].astype("string").str.strip()
            )
            output["mapping_update_date"] = pd.to_datetime(
                output["updateDate"], errors="coerce"
            ).dt.normalize()
            output["as_of_date"] = pd.Timestamp(as_of)
            output["retrieved_at"] = pd.Timestamp(retrieved_at)
            output["source"] = "baostock"
            output["request_id"] = snapshot.request_id
            return output[
                [
                    "as_of_date",
                    "symbol",
                    "stock_name",
                    "industry_name",
                    "industry_classification",
                    "mapping_update_date",
                    "source",
                    "retrieved_at",
                    "request_id",
                ]
            ].sort_values("symbol", ignore_index=True)

        return self._with_session(query)

    def is_trading_day(self, target: date) -> bool:
        calendar = self.fetch_trade_calendar(target, target)
        return bool(calendar.iloc[0]["is_trading_day"])

    def next_trading_day(self, target: date) -> date:
        calendar = self.fetch_trade_calendar(target + timedelta(days=1), target + timedelta(days=15))
        trading = calendar[calendar["is_trading_day"]]
        if trading.empty:
            raise BaostockError(f"No trading day found in 15 calendar days after {target}")
        return pd.Timestamp(trading.iloc[0]["calendar_date"]).date()

    def fetch_trade_calendar(self, start: date, end: date) -> pd.DataFrame:
        def query(bs: Any) -> pd.DataFrame:
            params = {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            }
            result = bs.query_trade_dates(**params)
            if result.error_code != "0":
                raise BaostockError(result.error_msg)
            raw = self._result_frame(result)
            retrieved_at = datetime.now(UTC)
            self.archive.save(
                "baostock",
                "query_trade_dates",
                raw,
                params,
                getattr(bs, "__version__", "unknown"),
                retrieved_at,
            )
            if raw.empty or not {"calendar_date", "is_trading_day"}.issubset(raw.columns):
                raise BaostockError(
                    f"Trade calendar contract failed for {start}..{end}: "
                    f"columns={list(raw.columns)}, rows={len(raw)}"
                )
            output = raw[["calendar_date", "is_trading_day"]].copy()
            output["calendar_date"] = pd.to_datetime(output["calendar_date"]).dt.normalize()
            output["is_trading_day"] = output["is_trading_day"].astype(str).str.strip().eq("1")
            return output.sort_values("calendar_date").reset_index(drop=True)

        return self._with_session(query)
