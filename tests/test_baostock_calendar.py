from datetime import date

import pytest

from quant.data.baostock_adapter import BaostockAdapter, BaostockError
from quant.data.raw_archive import RawArchive


class _Response:
    error_code = "0"
    error_msg = ""
    fields = ["calendar_date", "is_trading_day"]

    def __init__(self, rows: list[list[str]], fields: list[str] | None = None) -> None:
        self.rows = rows
        if fields is not None:
            self.fields = fields
        self._read = False
        self._index = -1

    def next(self) -> bool:
        self._index += 1
        if self._index >= len(self.rows):
            return False
        self._read = True
        return True

    def get_row_data(self) -> list[str]:
        return self.rows[self._index]


class _Login:
    error_code = "0"
    error_msg = ""


class _Baostock:
    __version__ = "test"

    def __init__(self, trading: str) -> None:
        self.trading = trading
        self.login_count = 0
        self.logout_count = 0

    def login(self) -> _Login:
        self.login_count += 1
        return _Login()

    def logout(self) -> None:
        self.logout_count += 1
        return None

    def query_trade_dates(self, **kwargs) -> _Response:
        start = date.fromisoformat(kwargs["start_date"])
        end = date.fromisoformat(kwargs["end_date"])
        rows: list[list[str]] = []
        current = start
        while current <= end:
            is_trading = self.trading if current.weekday() < 5 else "0"
            rows.append([current.isoformat(), is_trading])
            current = date.fromordinal(current.toordinal() + 1)
        return _Response(rows)

    def query_stock_industry(self, **kwargs) -> _Response:
        return _Response(
            [
                [
                    kwargs["date"],
                    "sh.600000",
                    "浦发银行",
                    "J66货币金融服务",
                    "证监会行业分类",
                ]
            ],
            [
                "updateDate",
                "code",
                "code_name",
                "industry",
                "industryClassification",
            ],
        )


class _ExpiredOnceBaostock(_Baostock):
    def __init__(self) -> None:
        super().__init__("1")
        self.expired = True

    def query_trade_dates(self, **kwargs) -> _Response:
        if self.expired:
            self.expired = False
            response = _Response([])
            response.error_code = "10001001"
            response.error_msg = "用户未登录"
            return response
        return super().query_trade_dates(**kwargs)


def test_baostock_trade_calendar_contract_and_archive(tmp_path) -> None:
    adapter = BaostockAdapter(RawArchive(tmp_path))
    adapter._module = lambda: _Baostock("1")  # type: ignore[method-assign]

    assert adapter.is_trading_day(date(2026, 7, 27))
    assert list((tmp_path / "baostock").rglob("*.json"))


def test_baostock_non_trading_day(tmp_path) -> None:
    adapter = BaostockAdapter(RawArchive(tmp_path))
    adapter._module = lambda: _Baostock("0")  # type: ignore[method-assign]

    assert not adapter.is_trading_day(date(2026, 7, 27))


def test_baostock_next_trading_day_skips_weekend(tmp_path) -> None:
    adapter = BaostockAdapter(RawArchive(tmp_path))
    adapter._module = lambda: _Baostock("1")  # type: ignore[method-assign]

    assert adapter.next_trading_day(date(2026, 7, 31)) == date(2026, 8, 3)


def test_baostock_explicit_session_reuses_login(tmp_path) -> None:
    module = _Baostock("1")
    adapter = BaostockAdapter(RawArchive(tmp_path))
    adapter._module = lambda: module  # type: ignore[method-assign]

    with adapter.session():
        assert adapter.is_trading_day(date(2026, 7, 27))
        assert adapter.is_trading_day(date(2026, 7, 28))

    assert module.login_count == 1
    assert module.logout_count == 1


def test_baostock_expired_session_reconnects_before_caller_retry(tmp_path) -> None:
    module = _ExpiredOnceBaostock()
    adapter = BaostockAdapter(RawArchive(tmp_path))
    adapter._module = lambda: module  # type: ignore[method-assign]

    with adapter.session():
        with pytest.raises(BaostockError, match="用户未登录"):
            adapter.fetch_trade_calendar(date(2026, 7, 27), date(2026, 7, 27))
        recovered = adapter.fetch_trade_calendar(date(2026, 7, 27), date(2026, 7, 27))

    assert bool(recovered.iloc[0]["is_trading_day"])
    assert module.login_count == 2
    assert module.logout_count == 2


def test_baostock_industry_contract_and_normalization(tmp_path) -> None:
    adapter = BaostockAdapter(RawArchive(tmp_path))
    adapter._module = lambda: _Baostock("1")  # type: ignore[method-assign]

    result = adapter.fetch_stock_industry(date(2026, 7, 27))

    assert result.iloc[0]["symbol"] == "SH600000"
    assert result.iloc[0]["stock_name"] == "浦发银行"
    assert result.iloc[0]["industry_name"] == "J66货币金融服务"
    assert result.iloc[0]["industry_classification"] == "证监会行业分类"
    assert list((tmp_path / "baostock").rglob("*.json"))
