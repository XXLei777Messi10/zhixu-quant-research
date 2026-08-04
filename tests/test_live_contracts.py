from datetime import date

import pytest

from quant.data.akshare_adapter import AKShareAdapter
from quant.data.baostock_adapter import BaostockAdapter
from quant.data.raw_archive import RawArchive


@pytest.mark.live
def test_baostock_historical_membership_contract(tmp_path) -> None:
    result = BaostockAdapter(RawArchive(tmp_path)).fetch_historical_csi300(date(2010, 1, 4))
    assert len(result) == 300
    assert {"updateDate", "code", "code_name", "symbol", "trade_date"}.issubset(result.columns)


@pytest.mark.live
def test_baostock_point_in_time_industry_contract(tmp_path) -> None:
    result = BaostockAdapter(RawArchive(tmp_path)).fetch_stock_industry(date(2026, 7, 27))
    assert len(result) >= 5_000
    assert {
        "as_of_date",
        "symbol",
        "stock_name",
        "industry_name",
        "industry_classification",
        "mapping_update_date",
    }.issubset(result.columns)
    assert result["mapping_update_date"].max() <= result["as_of_date"].max()


@pytest.mark.live
def test_akshare_stock_and_index_history_contracts(tmp_path) -> None:
    adapter = AKShareAdapter(RawArchive(tmp_path), requests_per_second=0)
    stock = adapter.fetch_stock_bars("SZ000001", date(2025, 1, 2), date(2025, 1, 10), "raw")
    index = adapter.fetch_index_bars("SH000300", date(2025, 1, 2), date(2025, 1, 10))
    required = {
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "source",
        "request_id",
    }
    assert not stock.empty and not index.empty
    assert required.issubset(stock.columns)
    assert required.issubset(index.columns)
    assert stock["symbol"].eq("SZ000001").all()
    assert index["symbol"].eq("SH000300").all()
    assert (stock[["open", "high", "low", "close", "volume"]] > 0).all().all()


@pytest.mark.live
def test_akshare_current_csi300_contract(tmp_path) -> None:
    result = AKShareAdapter(RawArchive(tmp_path), requests_per_second=0).fetch_current_csi300()
    code_column = "成分券代码" if "成分券代码" in result else "品种代码"
    assert len(result) >= 250
    assert code_column in result
