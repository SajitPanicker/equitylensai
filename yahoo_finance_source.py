"""Yahoo Finance data access for Equity Lens.

This module intentionally keeps Yahoo Finance retrieval separate from the RAG pipeline
and from any proprietary exchange filing logic. The integration uses yfinance, which is
an unofficial community-maintained interface to Yahoo's public finance endpoints and is
not an official exchange or government data source. Business, redistribution, or
commercial use should be reviewed against the yfinance package license and the terms of
any deployed environment before shipping externally.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - handled at runtime when the package is absent.
    yf = None


INDIAN_COMPANY_TICKERS: dict[str, str] = {
    "infosys": "INFY.NS",
    "infosys limited": "INFY.NS",
    "tcs": "TCS.NS",
    "tata consultancy services": "TCS.NS",
    "reliance": "RELIANCE.NS",
    "reliance industries": "RELIANCE.NS",
    "hdfc bank": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "wipro": "WIPRO.NS",
    "ltimindtree": "LTIM.NS",
    "ltimindtree limited": "LTIM.NS",
}


def _normalize_company_name(name: str) -> str:
    return " ".join(str(name or "").strip().lower().split())


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("%", "")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def resolve_ticker(company_name: str, ticker: str | None = None) -> str:
    """Resolve a company to a validated ticker without letting the LLM invent symbols."""

    if ticker:
        clean_ticker = ticker.strip().upper()
        if clean_ticker.endswith(".NS") or clean_ticker.endswith(".BO"):
            return clean_ticker
        return f"{clean_ticker}.NS"

    normalized = _normalize_company_name(company_name)
    if not normalized:
        raise ValueError("A company name or ticker is required to resolve Yahoo Finance data.")

    if normalized in INDIAN_COMPANY_TICKERS:
        return INDIAN_COMPANY_TICKERS[normalized]

    for alias, symbol in INDIAN_COMPANY_TICKERS.items():
        if alias in normalized or normalized in alias:
            return symbol

    raise ValueError(
        f"No deterministic Yahoo Finance ticker could be resolved for '{company_name}'. "
        "Use an explicit, supported Indian company name or a known ticker."
    )


def _safe_dict(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        return data
    return {}


def _as_kv(mapping: dict[str, Any], *keys: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in keys:
        if key in mapping:
            result[key] = mapping[key]
    return result


def _history_summary(history: Any, *, period: str = "1mo", interval: str = "1d") -> dict[str, Any]:
    if history is None or getattr(history, "empty", True):
        return {}
    rows: list[dict[str, Any]] = []
    for timestamp, row in history.iterrows():
        rows.append(
            {
                "date": str(timestamp.date()),
                "open": _coerce_float(row.get("Open")),
                "high": _coerce_float(row.get("High")),
                "low": _coerce_float(row.get("Low")),
                "close": _coerce_float(row.get("Close")),
                "adj_close": _coerce_float(row.get("Adj Close")),
                "volume": _coerce_float(row.get("Volume")),
            }
        )
    return {"period": period, "interval": interval, "points": rows[-250:]}


def _historical_returns(points: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Calculate interval and calendar-year price returns from normalized closes."""

    ordered_points = sorted(points, key=lambda point: str(point.get("date") or ""))
    interval_returns: list[dict[str, Any]] = []
    yearly_closes: dict[int, tuple[str, float, str]] = {}
    previous_close: float | None = None
    for point in ordered_points:
        adjusted_close = _coerce_float(point.get("adj_close"))
        close = adjusted_close if adjusted_close is not None else _coerce_float(point.get("close"))
        basis = "adj_close" if adjusted_close is not None else "close"
        point_date = str(point.get("date") or "")
        try:
            year = int(point_date[:4])
        except ValueError:
            continue
        if close is None:
            continue
        if previous_close not in (None, 0):
            interval_returns.append(
                {
                    "date": point_date,
                    "return_pct": ((close - previous_close) / previous_close) * 100,
                    "basis": basis,
                }
            )
        previous_close = close
        yearly_closes[year] = (point_date, close, basis)

    annual_returns: list[dict[str, Any]] = []
    for year, (point_date, close, basis) in sorted(yearly_closes.items()):
        previous = yearly_closes.get(year - 1)
        annual_returns.append(
            {
                "year": year,
                "through_date": point_date,
                "reference_price": close,
                "basis": basis,
                "return_pct": (
                    ((close - previous[1]) / previous[1]) * 100
                    if previous and previous[1] != 0
                    else None
                ),
                "comparison_available": previous is not None and previous[1] != 0,
            }
        )
    return {"interval_returns": interval_returns, "annual_returns": annual_returns}


def _annual_income_statement_trends(statement: Any) -> list[dict[str, Any]]:
    """Extract a small serializable set of annual financial figures from yfinance."""

    if statement is None or getattr(statement, "empty", True):
        return []
    aliases = {
        "revenue": ("total revenue", "revenue"),
        "net_income": ("net income", "net income common stockholders"),
        "operating_income": ("operating income",),
        "gross_profit": ("gross profit",),
        "ebitda": ("ebitda",),
    }
    rows_by_name = {
        str(index).strip().casefold(): row
        for index, row in statement.iterrows()
    }
    trends: list[dict[str, Any]] = []
    columns = sorted(statement.columns, key=str)[-5:]
    for column in columns:
        period = column.date().isoformat() if hasattr(column, "date") else str(column)
        record: dict[str, Any] = {"period": period}
        for metric, row_names in aliases.items():
            row = next(
                (rows_by_name[name] for name in row_names if name in rows_by_name),
                None,
            )
            value = _coerce_float(row.get(column)) if row is not None else None
            if value is not None:
                record[metric] = value
        if len(record) > 1:
            trends.append(record)
    return trends


def _financial_trend_changes(trends: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calculate annual revenue and net-income changes where adjacent periods exist."""

    calculated: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for trend in trends:
        item: dict[str, Any] = {"period": trend["period"]}
        for metric in ("revenue", "net_income"):
            current_value = trend.get(metric)
            previous_value = previous.get(metric) if previous else None
            if (
                isinstance(current_value, (int, float))
                and isinstance(previous_value, (int, float))
                and previous_value != 0
            ):
                item[f"{metric}_change_pct"] = (
                    (current_value - previous_value) / previous_value
                ) * 100
        calculated.append(item)
        previous = trend
    return calculated


def fetch_yahoo_finance_data(
    company_name: str,
    ticker: str | None = None,
    *,
    period: str = "1mo",
    interval: str = "1d",
    include_history: bool = False,
) -> dict[str, Any]:
    """Return a normalized subset of Yahoo Finance market and company data."""

    if yf is None:
        raise ImportError("yfinance is not installed. Add yfinance to requirements and install dependencies.")

    resolved_ticker = resolve_ticker(company_name, ticker)
    stock = yf.Ticker(resolved_ticker)
    info = _safe_dict(getattr(stock, "info", {}) or {})
    fast_info = _safe_dict(getattr(stock, "fast_info", {}) or {})

    market_data: dict[str, Any] = {
        "company_name": company_name,
        "ticker": resolved_ticker,
        "source": "Yahoo Finance via yfinance",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "latest_available_price": _coerce_float(info.get("regularMarketPrice") or fast_info.get("last_price")),
        "previous_close": _coerce_float(info.get("regularMarketPreviousClose") or fast_info.get("previous_close")),
        "day_change": _coerce_float(info.get("regularMarketChange") or fast_info.get("last_price") and info.get("regularMarketPreviousClose") and ((info.get("regularMarketPrice") or 0) - (info.get("regularMarketPreviousClose") or 0))),
        "market_cap": _coerce_float(info.get("marketCap") or fast_info.get("market_cap")),
        "pe_ratio": _coerce_float(info.get("trailingPE") or fast_info.get("pe_ratio")),
        "eps_trailing_twelve_months": _coerce_float(info.get("trailingEps") or info.get("epsTrailingTwelveMonths")),
        "dividend_yield": _coerce_float(info.get("dividendYield") or info.get("yield")),
        "currency": info.get("currency") or "INR",
        "exchange": info.get("exchange") or "NSE",
        "currency_symbol": info.get("currencySymbol") or "₹",
    }

    summary = {
        "market_cap": _coerce_float(info.get("marketCap") or fast_info.get("market_cap")),
        "enterprise_value": _coerce_float(info.get("enterpriseValue")),
        "pe_ratio": _coerce_float(info.get("trailingPE") or fast_info.get("pe_ratio")),
        "price_to_book": _coerce_float(info.get("priceToBook") or info.get("price_to_book")),
        "eps": _coerce_float(info.get("trailingEps") or info.get("epsTrailingTwelveMonths")),
        "dividend_yield": _coerce_float(info.get("dividendYield") or info.get("yield")),
        "return_on_equity": _coerce_float(info.get("returnOnEquity")),
        "profit_margin": _coerce_float(info.get("profitMargins")),
        "revenue": _coerce_float(info.get("totalRevenue") or info.get("totalRevenue")),
        "net_income": _coerce_float(info.get("netIncomeToCommon")),
        "free_cash_flow": _coerce_float(info.get("freeCashflow")),
    }

    financials = {
        "summary": summary,
        "income_statement": _as_kv(info, "totalRevenue", "netIncomeToCommon", "operatingCashflow", "ebitda"),
        "balance_sheet": _as_kv(info, "totalCash", "totalDebt", "totalAssets", "totalLiabilities"),
        "cash_flow": _as_kv(info, "freeCashflow", "operatingCashflow", "capitalExpenditure"),
    }

    result: dict[str, Any] = {
        "company_name": company_name,
        "ticker": resolved_ticker,
        "ticker_validated": True,
        "source": "Yahoo Finance via yfinance",
        "data_timestamp": datetime.now(timezone.utc).isoformat(),
        "latest_market_data": market_data,
        "financial_summary": financials,
        "notes": [
            "Latest market values are the most recent available Yahoo Finance values, not guaranteed real-time exchange prices.",
            "Yahoo Finance is an unofficial third-party data source; verify commercial, redistribution, and legal use requirements before deployment.",
        ],
    }

    if include_history:
        try:
            history = stock.history(period=period, interval=interval)
            historical_prices = _history_summary(history, period=period, interval=interval)
            result["historical_prices"] = historical_prices
            result["historical_returns"] = _historical_returns(
                historical_prices.get("points", [])
            )
        except Exception as error:  # pragma: no cover - graceful degradation when the endpoint is unavailable.
            result["historical_prices"] = {"period": period, "interval": interval, "points": []}
            result["historical_returns"] = {"interval_returns": [], "annual_returns": []}
            result["historical_data_error"] = f"{type(error).__name__}: {error}"

    try:
        annual_financial_trends = _annual_income_statement_trends(
            getattr(stock, "income_stmt", None)
        )
        result["annual_financial_trends"] = annual_financial_trends
        result["calculated_financial_metrics"] = {
            "annual_revenue_profit_changes": _financial_trend_changes(
                annual_financial_trends
            )
        }
    except Exception as error:  # pragma: no cover - optional yfinance data.
        result["annual_financial_trends"] = []
        result["calculated_financial_metrics"] = {
            "annual_revenue_profit_changes": []
        }
        result["financial_trends_error"] = f"{type(error).__name__}: {error}"

    return result


def fetch_latest_market_data(company_name: str, ticker: str | None = None) -> dict[str, Any]:
    return fetch_yahoo_finance_data(company_name, ticker, include_history=False)


def fetch_historical_prices(
    company_name: str,
    ticker: str | None = None,
    *,
    period: str = "1mo",
    interval: str = "1d",
) -> dict[str, Any]:
    return fetch_yahoo_finance_data(company_name, ticker, period=period, interval=interval, include_history=True)
