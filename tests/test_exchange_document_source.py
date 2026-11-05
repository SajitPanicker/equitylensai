"""Offline tests for official exchange filing adapters and download validation."""

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from exchange_document_source import (
    BSE_API_URL,
    DOCUMENT_CATEGORIES,
    NSE_API_URL,
    NSE_ANNUAL_REPORTS_API_URL,
    NSE_COMPANY_API_URL,
    NSE_FINANCIAL_RESULTS_API_URL,
    NSE_SEARCH_URL,
    BSEAdapter,
    ExchangeSourceError,
    ExchangeUnavailable,
    FilingRecord,
    NSEAdapter,
    classify_category,
    load_search_cache,
    load_document_catalog,
    official_url,
    save_document_catalog,
    save_search_cache,
)


def response(url, payload=None, content=b"", status=200, content_type="application/json"):
    result = Mock()
    result.url = url
    result.status_code = status
    result.headers = {"content-type": content_type}
    result.content = content
    result.is_redirect = False
    result.is_permanent_redirect = False
    result.close = Mock()
    result.iter_content.return_value = [content] if content else []
    result.json.return_value = payload
    result.raise_for_status.side_effect = (
        requests.HTTPError(response=Mock(status_code=status)) if status >= 400 else None
    )
    return result


def nse_company_response(symbol="INFY"):
    payload = {"symbol": symbol, "companyName": "Infosys Limited"} if symbol else {}
    return response(NSE_COMPANY_API_URL, payload)


class ExchangeDocumentSourceTests(unittest.TestCase):
    def test_nse_search_maps_fields_and_applies_pagination(self):
        rows = [
            {
                "symbol": "INFY",
                "sm_name": "Infosys Limited",
                "desc": "Financial Results",
                "attchmntText": "Unaudited financial results Q1 FY2026",
                "an_dt": "04-Nov-2026 20:26:34",
                "attchmntFile": "https://nsearchives.nseindia.com/corporate/infosys.pdf",
                "fileSize": "1.42 MB",
                "seq_id": "12345",
            },
            {
                "symbol": "INFY",
                "sm_name": "Infosys Limited",
                "desc": "Investor Presentation",
                "attchmntText": "Q1 FY2026 investor presentation",
                "an_dt": "04-Nov-2026 19:00:00",
                "attchmntFile": "https://nsearchives.nseindia.com/corporate/infosys.pptx",
                "seq_id": "12346",
            },
            {
                "symbol": "INFY",
                "sm_name": "Infosys Limited",
                "desc": "Board Meeting",
                "attchmntText": "Board meeting outcome",
                "an_dt": "04-Nov-2026 18:00:00",
                "attchmntFile": "https://nsearchives.nseindia.com/corporate/board.pdf",
                "seq_id": "12347",
            },
        ]
        adapter = NSEAdapter(min_interval=0)
        adapter.session.get = Mock(
            side_effect=[
                response(NSE_SEARCH_URL),
                nse_company_response(),
                response(NSE_API_URL, rows),
            ]
        )

        records, total = adapter.search(
            "INFY",
            date(2026, 11, 1),
            date(2026, 11, 4),
            limit=1,
            offset=1,
        )

        self.assertEqual(total, 3)
        self.assertEqual(len(records), 1)
        self.assertIn("investor presentation", records[0].title.casefold())
        adapter.session.get.assert_any_call(
            NSE_API_URL,
            params={
                "index": "equities",
                "symbol": "INFY",
                "from_date": "01-11-2026",
                "to_date": "04-11-2026",
            },
            timeout=(5.0, 25.0),
            allow_redirects=False,
            stream=False,
        )

    def test_nse_search_returns_exact_symbol_and_category_records(self):
        row = {
            "symbol": "INFY",
            "sm_name": "Infosys Limited",
            "desc": "Financial Results",
            "attchmntText": "Financial results for Q1 FY2026",
            "an_dt": "04-Nov-2026 20:26:34",
            "attchmntFile": "https://nsearchives.nseindia.com/corporate/infosys.pdf",
            "seq_id": "12345",
        }
        adapter = NSEAdapter(min_interval=0)
        adapter.session.get = Mock(
            side_effect=[response(NSE_SEARCH_URL), nse_company_response(), response(NSE_API_URL, [row])]
        )

        records, total = adapter.search("infy", date(2026, 11, 1), date(2026, 11, 4))

        self.assertEqual(total, 1)
        self.assertEqual(records[0].company, "Infosys Limited")
        self.assertEqual(records[0].ticker, "INFY")
        self.assertEqual(records[0].original_category, "Financial Results")
        self.assertEqual(records[0].publication_date, "2026-11-04")
        self.assertEqual(records[0].fiscal_year, "FY2026")
        self.assertEqual(records[0].quarter, "Q1")
        self.assertEqual(adapter.resolved_company, "Infosys Limited")

    def test_nse_empty_response_is_an_empty_page(self):
        adapter = NSEAdapter(min_interval=0)
        adapter.session.get = Mock(
            side_effect=[response(NSE_SEARCH_URL), nse_company_response(), response(NSE_API_URL, [])]
        )

        records, total = adapter.search("INFY", date(2026, 1, 1), date(2026, 1, 2))

        self.assertEqual(records, [])
        self.assertEqual(total, 0)

    def test_nse_category_and_subject_filters(self):
        rows = [
            {
                "symbol": "INFY",
                "sm_name": "Infosys Limited",
                "desc": "Investor Presentation",
                "attchmntText": "Q1 FY2026 investor presentation",
                "attchmntFile": "https://nsearchives.nseindia.com/infy.pptx",
            },
            {
                "symbol": "INFY",
                "sm_name": "Infosys Limited",
                "desc": "Financial Results",
                "attchmntText": "Q1 FY2026 financial results",
                "attchmntFile": "https://nsearchives.nseindia.com/infy.pdf",
            },
        ]
        adapter = NSEAdapter(min_interval=0)
        adapter.session.get = Mock(
            side_effect=[response(NSE_SEARCH_URL), nse_company_response(), response(NSE_API_URL, rows)]
        )

        records, total = adapter.search(
            "INFY",
            date(2026, 1, 1),
            date(2026, 12, 31),
            subject="presentation",
            categories=["Investor presentations"],
        )

        self.assertEqual(total, 1)
        self.assertEqual(records[0].file_format, ".pptx")

    def test_nse_annual_reports_use_the_official_dedicated_catalogue(self):
        annual_report = {
            "companyName": "Infosys Limited",
            "fromYr": "2025",
            "toYr": "2026",
            "submission_type": "Revised",
            "broadcast_dttm": "30-MAY-2026 20:04:13",
            "fileName": "https://nsearchives.nseindia.com/annual_reports/INFY_FY2026.pdf",
            "attFileSize": "8.57 MB",
        }
        adapter = NSEAdapter(min_interval=0)
        adapter.session.get = Mock(
            side_effect=[
                response(NSE_SEARCH_URL),
                nse_company_response(),
                response(NSE_API_URL, []),
                response(NSE_ANNUAL_REPORTS_API_URL, {"data": [annual_report]}),
            ]
        )

        records, total = adapter.search(
            "INFY",
            date(2026, 1, 1),
            date(2026, 12, 31),
            categories=["Annual reports"],
        )

        self.assertEqual(total, 1)
        self.assertEqual(records[0].original_category, "Annual Reports")
        self.assertEqual(records[0].fiscal_year, "FY2026")
        self.assertEqual(records[0].publication_date, "2026-05-30")
        self.assertEqual(records[0].file_format, ".pdf")
        self.assertIn("Revised", records[0].title)
        self.assertEqual(
            adapter.session.get.call_args_list[-1].args[0],
            NSE_ANNUAL_REPORTS_API_URL,
        )

    def test_nse_financial_results_keep_reporting_period_basis_and_xbrl_format(self):
        result_row = {
            "symbol": "INFY",
            "companyName": "Infosys Limited",
            "financialYear": "01-Apr-2024 To 31-Mar-2025",
            "relatingTo": "Third Quarter",
            "period": "Quarterly",
            "filingDate": "16-Jan-2025 19:42",
            "audited": "Audited",
            "consolidated": "Consolidated",
            "seqNumber": "1189815",
            "xbrl": "https://nsearchives.nseindia.com/corporate/xbrl/result.xml",
            "resultDetailedDataLink": None,
        }
        adapter = NSEAdapter(min_interval=0)
        adapter.session.get = Mock(
            side_effect=[
                response(NSE_SEARCH_URL),
                nse_company_response(),
                response(NSE_API_URL, []),
                response(NSE_FINANCIAL_RESULTS_API_URL, [result_row]),
                response(NSE_FINANCIAL_RESULTS_API_URL, []),
                response(NSE_FINANCIAL_RESULTS_API_URL, []),
            ]
        )

        records, total = adapter.search(
            "INFY",
            date(2024, 4, 1),
            date(2025, 3, 31),
            categories=["Financial results"],
        )

        self.assertEqual(total, 1)
        self.assertEqual(records[0].fiscal_year, "FY2025")
        self.assertEqual(records[0].quarter, "Q3")
        self.assertEqual(records[0].publication_date, "2025-01-16")
        self.assertEqual(records[0].file_format, ".xml")
        self.assertIn("Consolidated", records[0].description)

    def test_nse_rejects_unresolved_company_symbol_before_announcement_search(self):
        adapter = NSEAdapter(min_interval=0)
        adapter.session.get = Mock(side_effect=[response(NSE_SEARCH_URL), nse_company_response("")])

        with self.assertRaisesRegex(ValueError, "did not resolve"):
            adapter.search("NOTREAL", date(2026, 1, 1), date(2026, 1, 2))

        self.assertEqual(adapter.session.get.call_count, 2)

    def test_bse_search_maps_results_and_uses_page_number(self):
        row = {
            "SCRIP_CD": "500209",
            "SLONGNAME": "Infosys Limited",
            "NEWSSUB": "Investor Presentation for Q1 FY2026",
            "CATEGORYNAME": "Company Update",
            "NEWS_DT": "04-Nov-2026 20:00:00",
            "ATTACHMENTNAME": "https://www.bseindia.com/xml-data/corpfiling/attachment.pdf",
        }
        adapter = BSEAdapter(min_interval=0)
        adapter.session.get = Mock(return_value=response(BSE_API_URL, [row]))

        records, total = adapter.search(
            "500209", date(2026, 1, 1), date(2026, 12, 31), limit=25, offset=25
        )

        self.assertEqual(total, 1)
        self.assertEqual(records[0].company, "Infosys Limited")
        self.assertEqual(records[0].ticker, "500209")
        self.assertEqual(records[0].original_category, "Company Update")
        self.assertEqual(records[0].publication_date, "2026-11-04")
        self.assertEqual(adapter.session.get.call_args.kwargs["params"]["pageno"], 2)

    def test_bse_403_surfaces_official_search_fallback_without_retry(self):
        adapter = BSEAdapter(min_interval=0)
        denied = response(BSE_API_URL, status=403)
        adapter.session.get = Mock(return_value=denied)

        with self.assertRaises(ExchangeUnavailable) as caught:
            adapter.search("500209", date(2026, 1, 1), date(2026, 1, 2))

        self.assertIn("HTTP 403", str(caught.exception))
        self.assertEqual(caught.exception.search_url, adapter.search_url)
        adapter.session.get.assert_called_once()

    def test_bse_result_normalization_preserves_original_category_and_url(self):
        record = BSEAdapter._record(
            {
                "SCRIP_CD": "500209",
                "SLONGNAME": "Infosys Limited",
                "NEWSSUB": "Investor Presentation for Q1 FY2026",
                "CATEGORYNAME": "Company Update",
                "NEWS_DT": "04-Nov-2026 20:00:00",
                "ATTACHMENTNAME": "https://www.bseindia.com/xml-data/corpfiling/attachment.pdf",
            },
            "500209",
        )

        self.assertEqual(record.company, "Infosys Limited")
        self.assertEqual(record.ticker, "500209")
        self.assertEqual(record.original_category, "Company Update")
        self.assertEqual(record.fiscal_year, "FY2026")
        self.assertEqual(record.source_url, "https://www.bseindia.com/xml-data/corpfiling/attachment.pdf")

    def test_category_mapping_covers_supported_groups(self):
        examples = {
            "Financial results": "Unaudited financial results for quarter ended",
            "Annual reports": "Annual Report FY2025",
            "Investor presentations": "Investor Presentation",
            "Corporate announcements and material events": "Acquisition of material business",
            "Earnings calls and analyst meetings": "Earnings Call Transcript",
            "Shareholding patterns": "Shareholding Pattern",
            "Governance and compliance": "Corporate Governance Compliance Report",
            "Credit ratings": "Credit Rating Revision",
            "Board meetings": "Board Meeting Outcome",
            "Other company filings": "Certificate under filing regulation",
        }
        for expected, sample in examples.items():
            with self.subTest(category=expected):
                self.assertIn(expected, classify_category(sample))

    def test_unapproved_redirect_is_rejected_before_following(self):
        adapter = NSEAdapter(min_interval=0)
        redirect = response(NSE_API_URL, status=302)
        redirect.is_redirect = True
        redirect.headers = {"location": "https://example.invalid/filing.pdf"}
        adapter.session.get = Mock(return_value=redirect)

        with self.assertRaisesRegex(ExchangeSourceError, "redirected"):
            adapter._get(NSE_API_URL)

        adapter.session.get.assert_called_once()

    def test_rate_limit_retries_are_bounded(self):
        adapter = NSEAdapter(min_interval=0)
        limited = response(NSE_API_URL, status=429)
        adapter.session.get = Mock(return_value=limited)

        with patch("exchange_document_source.time.sleep") as sleep:
            with self.assertRaises(ExchangeSourceError):
                adapter._get(NSE_API_URL)

        self.assertEqual(adapter.session.get.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_invalid_or_non_official_urls_are_rejected(self):
        self.assertTrue(official_url("https://www.nseindia.com/path"))
        self.assertFalse(official_url("http://www.nseindia.com/path"))
        self.assertFalse(official_url("https://nseindia.com.attacker.invalid/path"))
        self.assertFalse(official_url("https://user@www.nseindia.com/path"))

        with self.assertRaises(ExchangeSourceError):
            NSEAdapter._record(
                {
                    "symbol": "INFY",
                    "attchmntFile": "https://example.invalid/filing.pdf",
                },
                "INFY",
            )

    def test_download_checks_signature_and_creates_persistent_cache_path(self):
        record = FilingRecord(
            exchange="NSE",
            company="Infosys Limited",
            ticker="INFY",
            title="Annual Report FY2026",
            original_category="Annual Report",
            publication_date="2026-11-04",
            source_url="https://nsearchives.nseindia.com/corporate/report.pdf",
            fiscal_year="FY2026",
        )
        adapter = NSEAdapter(min_interval=0)
        adapter.session.get = Mock(
            return_value=response(
                record.source_url,
                content=b"%PDF-1.7\nfile",
                content_type="application/pdf",
            )
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            downloaded = adapter.download(record, Path(temporary_directory))
            repeated = adapter.download(record, Path(temporary_directory))

            self.assertTrue(downloaded.path.is_file())
            self.assertEqual(downloaded.path.suffix, ".pdf")
            self.assertEqual(downloaded.file_hash, repeated.file_hash)
            self.assertTrue(repeated.skipped_existing)
            self.assertIn("nse", str(downloaded.path))
            self.assertIn("INFY", str(downloaded.path))

    def test_download_rejects_html_disguised_as_pdf(self):
        record = FilingRecord(
            exchange="NSE",
            company="Infosys Limited",
            ticker="INFY",
            title="Filing",
            original_category="Other",
            publication_date="",
            source_url="https://nsearchives.nseindia.com/corporate/file.pdf",
        )
        adapter = NSEAdapter(min_interval=0)
        adapter.session.get = Mock(
            return_value=response(
                record.source_url,
                content=b"<html>blocked</html>",
                content_type="application/octet-stream",
            )
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ExchangeSourceError, "signature"):
                adapter.download(record, Path(temporary_directory))

    def test_cache_round_trip_and_all_categories_are_available(self):
        record = FilingRecord(
            exchange="NSE",
            company="Infosys Limited",
            ticker="INFY",
            title="Annual report",
            original_category="Annual Report",
            publication_date="2026-11-04",
            source_url="https://nsearchives.nseindia.com/corporate/report.pdf",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "search.json"
            save_search_cache(cache_path, [record])
            self.assertEqual(load_search_cache(cache_path)[0], record)

            catalog_path = Path(temporary_directory) / "document_catalog.json"
            catalog = {"INFY|123": {"status": "indexed", "file_hash": "abc"}}
            save_document_catalog(catalog_path, catalog)
            self.assertEqual(load_document_catalog(catalog_path), catalog)

        self.assertGreaterEqual(len(DOCUMENT_CATEGORIES), 9)
        self.assertIn("Credit ratings", classify_category("Credit rating revised"))
        self.assertIn("Board meetings", classify_category("Board Meeting Outcome"))
        self.assertIn("Shareholding patterns", classify_category("Shareholding Pattern"))


if __name__ == "__main__":
    unittest.main()