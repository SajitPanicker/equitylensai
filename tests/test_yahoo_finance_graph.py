"""Offline tests for structured Yahoo Finance graph output."""

import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

from yahoo_finance_graph import build_yahoo_finance_graph
from yahoo_finance_source import (
    _financial_trend_changes,
    _historical_returns,
    fetch_yahoo_finance_data,
)


class FakeHistory:
    empty = False

    def iterrows(self):
        return iter(
            [
                (datetime(2024, 12, 31), {"Open": 90, "High": 101, "Low": 89, "Close": 100}),
                (datetime(2025, 12, 31), {"Open": 100, "High": 121, "Low": 99, "Close": 120}),
            ]
        )


class FakeIncomeStatement:
    empty = False
    columns = [datetime(2025, 3, 31)]

    def iterrows(self):
        return iter(
            [
                ("Total Revenue", {self.columns[0]: 500}),
                ("Net Income", {self.columns[0]: 75}),
            ]
        )


class YahooFinanceGraphTests(unittest.TestCase):
    def test_graph_returns_structured_market_data_for_parent_synthesis(self):
        market_data = {
            "company_name": "Infosys",
            "ticker": "INFY.NS",
            "latest_market_data": {"latest_available_price": 1500.0},
            "historical_prices": {"points": [{"date": "2025-12-31", "close": 120.0}]},
            "historical_returns": {"annual_returns": [{"year": 2025, "return_pct": 20.0}]},
            "financial_summary": {"summary": {"revenue": 500.0, "net_income": 75.0}},
            "source": "Yahoo Finance via yfinance",
            "data_timestamp": "2026-11-08T10:00:00+00:00",
        }
        graph = build_yahoo_finance_graph()

        with patch("yahoo_finance_graph.fetch_yahoo_finance_data", return_value=market_data) as fetch:
            result = graph.invoke(
                {
                    "question": "Compare historical trend for the last 5 years",
                    "company_name": "Infosys",
                    "ticker": "INFY.NS",
                }
            )

        self.assertEqual(result["market_data"], market_data)
        self.assertEqual(result["yahoo_result"], market_data)
        self.assertEqual(result["is_error"], False)
        fetch.assert_called_once_with(
            "Infosys",
            ticker="INFY.NS",
            period="5y",
            interval="1mo",
            include_history=True,
        )

    def test_graph_returns_explicit_error_and_no_fabricated_market_data(self):
        graph = build_yahoo_finance_graph()
        with patch(
            "yahoo_finance_graph.fetch_yahoo_finance_data",
            side_effect=RuntimeError("Yahoo unavailable"),
        ):
            result = graph.invoke({"question": "What is Infosys share price today"})

        self.assertTrue(result["is_error"])
        self.assertEqual(result["market_data"], {})
        self.assertIn("Yahoo unavailable", result["error_message"])

    def test_historical_returns_are_calculated_from_closes(self):
        returns = _historical_returns(
            [
                {"date": "2024-12-31", "close": 100.0},
                {"date": "2025-12-31", "close": 120.0},
                {"date": "2026-06-30", "close": 90.0},
            ]
        )

        self.assertEqual(returns["interval_returns"][0]["return_pct"], 20.0)
        self.assertEqual(returns["interval_returns"][1]["return_pct"], -25.0)
        self.assertIsNone(returns["annual_returns"][0]["return_pct"])
        self.assertEqual(returns["annual_returns"][1]["return_pct"], 20.0)
        self.assertEqual(returns["annual_returns"][2]["return_pct"], -25.0)

    def test_revenue_and_net_income_changes_are_calculated_deterministically(self):
        changes = _financial_trend_changes(
            [
                {"period": "FY2024", "revenue": 400.0, "net_income": 60.0},
                {"period": "FY2025", "revenue": 500.0, "net_income": 75.0},
            ]
        )

        self.assertEqual(changes[0], {"period": "FY2024"})
        self.assertEqual(changes[1]["revenue_change_pct"], 25.0)
        self.assertEqual(changes[1]["net_income_change_pct"], 25.0)

    def test_source_returns_bounded_serializable_history_and_financial_trends(self):
        stock = SimpleNamespace(
            info={
                "regularMarketPrice": 1500,
                "marketCap": 1000000,
                "trailingPE": 20,
                "trailingEps": 75,
                "totalRevenue": 500,
                "netIncomeToCommon": 75,
            },
            fast_info={},
            history=lambda **_kwargs: FakeHistory(),
            income_stmt=FakeIncomeStatement(),
        )

        with patch("yahoo_finance_source.yf", SimpleNamespace(Ticker=lambda _ticker: stock)):
            result = fetch_yahoo_finance_data(
                "Infosys",
                period="5y",
                interval="1mo",
                include_history=True,
            )

        self.assertEqual(len(result["historical_prices"]["points"]), 2)
        self.assertEqual(result["historical_returns"]["annual_returns"][1]["return_pct"], 20.0)
        self.assertEqual(
            result["annual_financial_trends"][0],
            {"period": "2025-03-31", "revenue": 500.0, "net_income": 75.0},
        )
        self.assertIsInstance(result["data_timestamp"], str)
        self.assertIn("Yahoo Finance", result["source"])
        self.assertNotIn("history", repr(result).casefold())


if __name__ == "__main__":
    unittest.main()
