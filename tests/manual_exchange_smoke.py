"""Optional live smoke test for official exchange search routes; not run by unittest."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exchange_document_source import (
    DOCUMENT_CATEGORIES,
    BSEAdapter,
    ExchangeSourceError,
    ExchangeUnavailable,
    NSEAdapter,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check live official exchange filing search access.")
    parser.add_argument("--exchange", choices=("NSE", "BSE"), required=True)
    parser.add_argument("--nse-symbol", help="Exact NSE symbol, for example INFY")
    parser.add_argument("--bse-code", help="Exact BSE numeric scrip code, for example 500209")
    parser.add_argument(
        "--category",
        action="append",
        choices=DOCUMENT_CATEGORIES,
        help="NSE category to exercise; may be supplied more than once",
    )
    parser.add_argument("--days", type=int, default=14, help="Search the last N publication days")
    args = parser.parse_args()

    if args.days < 1 or args.days > 3650:
        parser.error("--days must be between 1 and 3650")
    end_date = date.today()
    start_date = end_date - timedelta(days=args.days)

    try:
        if args.exchange == "NSE":
            if not args.nse_symbol:
                parser.error("--nse-symbol is required for NSE")
            adapter = NSEAdapter()
            records, total = adapter.search(
                args.nse_symbol,
                start_date,
                end_date,
                limit=10,
                categories=args.category,
            )
            print(f"NSE resolved: {args.nse_symbol.upper()} -> {adapter.resolved_company}")
        else:
            if not args.bse_code:
                parser.error("--bse-code is required for BSE")
            if args.category:
                parser.error("--category is currently available only for NSE smoke checks")
            records, total = BSEAdapter().search(args.bse_code, start_date, end_date, limit=10)
        print(f"{args.exchange}: {total} result(s), displaying {len(records)}")
        for record in records:
            print(f"{record.publication_date} | {record.ticker} | {record.original_category} | {record.title}")
            print(f"  {record.source_url or '(no downloadable attachment)'}")
        return 0
    except ExchangeUnavailable as error:
        print(f"{error}\nOfficial search: {error.search_url}", file=sys.stderr)
        return 2
    except (ExchangeSourceError, ValueError) as error:
        print(f"Search failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
