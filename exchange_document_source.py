"""Official exchange filing discovery and bounded local document downloads."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests


NSE_SEARCH_URL = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
NSE_RESULTS_URL = "https://www.nseindia.com/companies-listing/corporate-filings-financial-results"
NSE_ANNUAL_REPORTS_URL = "https://www.nseindia.com/companies-listing/corporate-filings-annual-reports"
NSE_SHAREHOLDING_URL = "https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern"
NSE_GOVERNANCE_URL = "https://www.nseindia.com/companies-listing/corporate-filings-governance"
NSE_BOARD_MEETINGS_URL = "https://www.nseindia.com/companies-listing/corporate-filings-board-meetings"
NSE_API_URL = "https://www.nseindia.com/api/corporate-announcements"
NSE_COMPANY_API_URL = "https://www.nseindia.com/api/equity-metainfo"
NSE_ANNUAL_REPORTS_API_URL = "https://www.nseindia.com/api/annual-reports"
NSE_FINANCIAL_RESULTS_API_URL = "https://www.nseindia.com/api/corporates-financial-results"
BSE_SEARCH_URL = "https://www.bseindia.com/corporates/ann"
BSE_RESULTS_URL = "https://www.bseindia.com/corporates/Comp_ResultsNew"
BSE_API_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"

MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
MAX_SEARCH_RESULTS = 500
SUPPORTED_FORMATS = {".pdf", ".pptx"}
ALLOWED_HOSTS = {
    "www.nseindia.com",
    "nsearchives.nseindia.com",
    "www.bseindia.com",
    "api.bseindia.com",
}

DOCUMENT_CATEGORIES = (
    "Financial results",
    "Annual reports",
    "Investor presentations",
    "Corporate announcements and material events",
    "Earnings calls and analyst meetings",
    "Shareholding patterns",
    "Governance and compliance",
    "Credit ratings",
    "Board meetings",
    "Other company filings",
)


class ExchangeSourceError(RuntimeError):
    """An exchange response or filing could not be used safely."""


class ExchangeUnavailable(ExchangeSourceError):
    """The official exchange search is available, but programmatic access is not."""

    def __init__(self, exchange: str, message: str, search_url: str):
        super().__init__(message)
        self.exchange = exchange
        self.search_url = search_url


@dataclass
class FilingRecord:
    exchange: str
    company: str
    ticker: str
    title: str
    original_category: str
    publication_date: str
    source_url: str
    reporting_period: str = ""
    fiscal_year: str = ""
    quarter: str = ""
    file_format: str = ""
    file_size: str = ""
    description: str = ""
    document_id: str = ""
    index_status: str = "Discovered"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class DownloadedFiling:
    path: Path
    file_hash: str
    file_format: str
    size_bytes: int
    skipped_existing: bool = False


def official_url(url: str, allowed_hosts: set[str] = ALLOWED_HOSTS) -> bool:
    """Accept only HTTPS URLs on the explicitly approved official exchange hosts."""

    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.hostname is not None
        and parsed.hostname.lower() in allowed_hosts
        and not parsed.username
        and not parsed.password
    )


def classify_category(text: str) -> list[str]:
    """Map exchange filing text to broad UI filters without replacing its source label."""

    value = text.casefold()
    patterns = {
        "Financial results": r"financial result|quarterly result|audited result|integrated filing.?financial|earnings",
        "Annual reports": r"annual report|annual accounts",
        "Investor presentations": r"investor presentation|investor deck|presentation",
        "Corporate announcements and material events": r"acquisition|merger|investment|order|litigation|expansion|material event|regulation 30|business update|management change",
        "Earnings calls and analyst meetings": r"earnings call|analyst|investor meet|conference call|transcript|audio call|recording",
        "Shareholding patterns": r"shareholding|promoter holding|ownership pattern",
        "Governance and compliance": r"governance|compliance|secretarial|regulation 27|regulation 34|regulation 33|related party",
        "Credit ratings": r"credit rating|rating action|rating rationale|rating revision",
        "Board meetings": r"board meeting|board outcome|board of directors",
    }
    matched = [category for category, pattern in patterns.items() if re.search(pattern, value)]
    return matched or ["Other company filings"]


def _safe_component(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned[:80] or fallback


def _date_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    import datetime

    for pattern in (
        "%d-%b-%Y %H:%M:%S",
        "%d-%b-%Y %H:%M",
        "%d-%b-%Y",
        "%d-%m-%Y",
        "%Y-%m-%d",
    ):
        try:
            return datetime.datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            pass
    return text


def _period_fields(text: str) -> tuple[str, str, str]:
    """Extract period labels only when an explicit financial year or quarter is stated."""

    fiscal_year = ""
    quarter = ""
    range_match = re.search(r"\bFY\s*((?:19|20)\d{2})\s*[-–/]\s*(\d{2,4})\b", text, re.IGNORECASE)
    if range_match:
        end_year = range_match.group(2)
        end_year = f"20{end_year}" if len(end_year) == 2 else end_year
        fiscal_year = f"FY{end_year}"
    else:
        year_match = re.search(r"\bFY\s*((?:19|20)\d{2})\b", text, re.IGNORECASE)
        if year_match:
            fiscal_year = f"FY{year_match.group(1)}"
    quarter_match = re.search(r"\bQ([1-4])\b", text, re.IGNORECASE)
    if quarter_match:
        quarter = f"Q{quarter_match.group(1)}"
    elif re.search(r"\bhalf[- ]year", text, re.IGNORECASE):
        quarter = "Half-year"
    elif re.search(r"\bannual\b", text, re.IGNORECASE):
        quarter = "Annual"
    reporting_period = " ".join(value for value in (fiscal_year, quarter) if value)
    return fiscal_year, quarter, reporting_period


def _format_from_url(url: str) -> str:
    return Path(urlparse(url).path).suffix.lower()


class _RateLimitedAdapter:
    exchange = ""
    search_url = ""
    _shared_rate_lock = threading.Lock()
    _last_request_by_host: dict[str, float] = {}

    def __init__(self, timeout: tuple[float, float] = (5.0, 25.0), min_interval: float = 1.0):
        self.timeout = timeout
        self.min_interval = min_interval
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "EquityLens/1.0 (official exchange disclosure research)",
                "Accept": "application/json, text/plain, */*",
            }
        )
        self.resolved_company = ""
    def _wait_turn(self, host: str) -> None:
        with self._shared_rate_lock:
            last_request = self._last_request_by_host.get(host, 0.0)
            delay = self.min_interval - (time.monotonic() - last_request)
            if delay > 0:
                time.sleep(delay)
            self._last_request_by_host[host] = time.monotonic()

    def _get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> requests.Response:
        if not official_url(url):
            raise ExchangeSourceError("Refused a non-official or non-HTTPS exchange URL.")
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                request_url = url
                request_params = params
                for _ in range(6):
                    host = urlparse(request_url).hostname or ""
                    self._wait_turn(host)
                    response = self.session.get(
                        request_url,
                        params=request_params,
                        timeout=self.timeout,
                        allow_redirects=False,
                        stream=stream,
                    )
                    request_params = None
                    if response.is_redirect or response.is_permanent_redirect:
                        location = response.headers.get("location", "")
                        request_url = urljoin(request_url, location)
                        if not official_url(request_url):
                            raise ExchangeSourceError("Exchange redirected to a non-approved host.")
                        continue
                    break
                else:
                    raise ExchangeSourceError("Exchange exceeded the permitted redirect count.")
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < 2:
                        response.close()
                        time.sleep(2**attempt)
                        continue
                response.raise_for_status()
                return response
            except requests.HTTPError as error:
                status = error.response.status_code if error.response is not None else 0
                if status == 403:
                    raise ExchangeUnavailable(
                        self.exchange,
                        f"{self.exchange} denied programmatic access (HTTP 403). Open its official search page instead.",
                        self.search_url,
                    ) from error
                if status != 429 and status < 500:
                    raise ExchangeSourceError(
                        f"{self.exchange} returned HTTP {status} for the official request."
                    ) from error
                last_error = error
            except (requests.Timeout, requests.ConnectionError) as error:
                last_error = error
            if attempt < 2:
                time.sleep(2**attempt)
        raise ExchangeSourceError(
            f"{self.exchange} request failed after bounded retries ({type(last_error).__name__})."
        ) from last_error

    def download(self, record: FilingRecord, cache_root: Path) -> DownloadedFiling:
        if record.exchange != self.exchange:
            raise ExchangeSourceError("The filing belongs to a different exchange adapter.")
        if not official_url(record.source_url):
            raise ExchangeSourceError("The filing URL is not on an approved official exchange host.")

        response = self._get(record.source_url, stream=True)
        try:
            content_type = response.headers.get("content-type", "").split(";", maxsplit=1)[0].strip().lower()
            if content_type in {"text/html", "application/json", "text/plain"}:
                raise ExchangeSourceError("The exchange returned a web page or data response instead of a document.")
            chunks: list[bytes] = []
            size_bytes = 0
            for block in response.iter_content(chunk_size=1024 * 1024):
                if not block:
                    continue
                size_bytes += len(block)
                if size_bytes > MAX_DOWNLOAD_BYTES:
                    raise ExchangeSourceError("The filing exceeds the 50 MB download limit.")
                chunks.append(block)
            content = b"".join(chunks)
            if not content:
                raise ExchangeSourceError("The exchange returned an empty document.")
        finally:
            response.close()

        extension = _format_from_url(response.url) or record.file_format.lower()
        if extension not in SUPPORTED_FORMATS:
            raise ExchangeSourceError(
                f"Unsupported filing format {extension or '(unknown)'}; only PDF and PowerPoint are indexed."
            )
        if extension == ".pdf" and not content.startswith(b"%PDF-"):
            raise ExchangeSourceError("The downloaded content is not a valid PDF signature.")
        if extension in {".ppt", ".pptx"} and not content.startswith(b"PK\x03\x04"):
            raise ExchangeSourceError("The downloaded content is not a valid PowerPoint ZIP signature.")

        digest = hashlib.sha256(content).hexdigest()
        company = _safe_component(record.ticker or record.company, "unknown-company")
        period = _safe_component(record.fiscal_year or "unclassified-period", "unclassified-period")
        category = _safe_component(record.original_category or "unclassified-category", "unclassified-category")
        filing_key = hashlib.sha256(
            (record.document_id or record.source_url).encode("utf-8")
        ).hexdigest()[:24]
        destination = cache_root / self.exchange.lower() / company / period / category / f"{filing_key}{extension}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == digest:
            return DownloadedFiling(destination, digest, extension, len(content), True)
        temporary_path = destination.with_suffix(destination.suffix + ".part")
        temporary_path.write_bytes(content)
        temporary_path.replace(destination)
        return DownloadedFiling(destination, digest, extension, size_bytes)


class NSEAdapter(_RateLimitedAdapter):
    exchange = "NSE"
    search_url = NSE_SEARCH_URL

    def search(
        self,
        symbol: str,
        from_date: date,
        to_date: date,
        limit: int = 100,
        offset: int = 0,
        subject: str = "",
        categories: list[str] | None = None,
    ) -> tuple[list[FilingRecord], int]:
        symbol = symbol.strip().upper()
        if not re.fullmatch(r"[A-Z0-9&.-]{1,30}", symbol):
            raise ValueError("Enter an exact NSE trading symbol; company-name matching is not inferred.")
        if from_date > to_date:
            raise ValueError("The publication start date must be on or before the end date.")
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("Result pages must be 1-100 records with a non-negative offset.")

        self._get(NSE_SEARCH_URL)
        company_response = self._get(NSE_COMPANY_API_URL, {"symbol": symbol})
        try:
            company_info = company_response.json()
        except (requests.JSONDecodeError, ValueError) as error:
            raise ExchangeSourceError("NSE returned an unexpected company-identity response.") from error
        if not isinstance(company_info, dict) or str(company_info.get("symbol", "")).upper() != symbol:
            raise ValueError(f"NSE did not resolve {symbol} to a listed company.")
        self.resolved_company = str(company_info.get("companyName") or "").strip()
        if not self.resolved_company:
            raise ExchangeSourceError("NSE resolved the symbol but did not return a company name.")
        response = self._get(
            NSE_API_URL,
            {
                "index": "equities",
                "symbol": symbol,
                "from_date": from_date.strftime("%d-%m-%Y"),
                "to_date": to_date.strftime("%d-%m-%Y"),
            },
        )
        try:
            rows = response.json()
        except (requests.JSONDecodeError, ValueError) as error:
            raise ExchangeSourceError("NSE returned an unexpected non-JSON filing response.") from error
        if not isinstance(rows, list):
            raise ExchangeSourceError("NSE returned an unexpected filing response shape.")

        records = [
            self._record(row, symbol)
            for row in rows
            if isinstance(row, dict)
            and str(row.get("symbol") or symbol).strip().upper() == symbol
        ]
        if categories and "Financial results" in categories:
            for period in ("Quarterly", "Half-Yearly", "Annual"):
                result_response = self._get(
                    NSE_FINANCIAL_RESULTS_API_URL,
                    {"index": "equities", "symbol": symbol, "period": period},
                )
                try:
                    result_rows = result_response.json()
                except (requests.JSONDecodeError, ValueError) as error:
                    raise ExchangeSourceError("NSE returned an unexpected financial-results response.") from error
                if not isinstance(result_rows, list):
                    raise ExchangeSourceError("NSE returned an unexpected financial-results response shape.")
                records.extend(
                    self._financial_result_record(row, symbol)
                    for row in result_rows
                    if isinstance(row, dict)
                    and str(row.get("symbol") or symbol).strip().upper() == symbol
                )
        if categories and "Annual reports" in categories:
            annual_response = self._get(
                NSE_ANNUAL_REPORTS_API_URL,
                {"index": "equities", "symbol": symbol},
            )
            try:
                annual_payload = annual_response.json()
            except (requests.JSONDecodeError, ValueError) as error:
                raise ExchangeSourceError("NSE returned an unexpected annual-report response.") from error
            annual_rows = annual_payload.get("data", []) if isinstance(annual_payload, dict) else []
            if not isinstance(annual_rows, list):
                raise ExchangeSourceError("NSE returned an unexpected annual-report response shape.")
            records.extend(
                self._annual_report_record(row, symbol)
                for row in annual_rows
                if isinstance(row, dict)
            )
        if subject.strip():
            term = subject.casefold().strip()
            records = [record for record in records if term in f"{record.title} {record.description}".casefold()]
        if categories:
            wanted = set(categories)
            records = [record for record in records if wanted.intersection(classify_category(f"{record.original_category} {record.title} {record.description}"))]
        records = [
            record
            for record in records
            if not record.publication_date
            or from_date.isoformat() <= record.publication_date <= to_date.isoformat()
        ]
        total = min(len(records), MAX_SEARCH_RESULTS)
        bounded = records[:MAX_SEARCH_RESULTS]
        return bounded[offset : offset + limit], total

    @staticmethod
    def _record(row: dict[str, Any], requested_symbol: str) -> FilingRecord:
        symbol = str(row.get("symbol") or requested_symbol).strip().upper()
        company = str(row.get("sm_name") or row.get("companyName") or "").strip()
        title = str(row.get("subject") or row.get("attchmntText") or row.get("desc") or "Exchange filing").strip()
        category = str(row.get("desc") or row.get("category") or "Other company filings").strip()
        source_url = str(row.get("attchmntFile") or row.get("attachmentFile") or "").strip()
        if source_url and not urlparse(source_url).scheme:
            source_url = urljoin("https://nsearchives.nseindia.com/", source_url.lstrip("/"))
        if source_url and not official_url(source_url):
            raise ExchangeSourceError("NSE returned a filing link outside its approved official hosts.")
        fiscal_year, quarter, reporting_period = _period_fields(f"{title} {category}")
        return FilingRecord(
            exchange="NSE",
            company=company,
            ticker=symbol,
            title=title,
            original_category=category,
            publication_date=_date_text(row.get("an_dt") or row.get("date")),
            source_url=source_url,
            reporting_period=reporting_period,
            fiscal_year=fiscal_year,
            quarter=quarter,
            file_format=_format_from_url(source_url),
            file_size=str(row.get("fileSize") or row.get("attFileSize") or ""),
            description=str(row.get("attchmntText") or "").strip(),
            document_id=str(row.get("seq_id") or row.get("id") or ""),
        )

    @staticmethod
    def _annual_report_record(row: dict[str, Any], symbol: str) -> FilingRecord:
        from_year = str(row.get("fromYr") or "").strip()
        to_year = str(row.get("toYr") or "").strip()
        submission_type = str(row.get("submission_type") or "").strip()
        title = "Annual Report"
        if from_year and to_year:
            title += f" ({from_year}-{to_year})"
        if submission_type and submission_type != "-":
            title += f" - {submission_type}"
        source_url = str(row.get("fileName") or "").strip()
        if source_url and not official_url(source_url):
            raise ExchangeSourceError("NSE returned an annual-report link outside its approved official hosts.")
        fiscal_year = f"FY{to_year}" if to_year.isdigit() else ""
        return FilingRecord(
            exchange="NSE",
            company=str(row.get("companyName") or "").strip(),
            ticker=symbol,
            title=title,
            original_category="Annual Reports",
            publication_date=_date_text(row.get("broadcast_dttm") or row.get("disseminationDateTime")),
            source_url=source_url,
            reporting_period=f"FY{from_year}-{to_year}" if from_year and to_year else "",
            fiscal_year=fiscal_year,
            quarter="Annual" if fiscal_year else "",
            file_format=_format_from_url(source_url),
            file_size=str(row.get("attFileSize") or ""),
            description="",
            document_id=source_url,
        )

    @staticmethod
    def _financial_result_record(row: dict[str, Any], symbol: str) -> FilingRecord:
        financial_year = str(row.get("financialYear") or "").strip()
        relating_to = str(row.get("relatingTo") or "").strip()
        period = str(row.get("period") or "").strip()
        fiscal_year_match = re.search(r"\b31-Mar-((?:19|20)\d{2})\b", financial_year, re.IGNORECASE)
        fiscal_year = f"FY{fiscal_year_match.group(1)}" if fiscal_year_match else ""
        quarter = ""
        quarter_match = re.search(r"\b(first|second|third|fourth) quarter\b", relating_to, re.IGNORECASE)
        if quarter_match:
            quarter = {
                "first": "Q1",
                "second": "Q2",
                "third": "Q3",
                "fourth": "Q4",
            }[quarter_match.group(1).casefold()]
        elif "half" in relating_to.casefold() or period.casefold() == "half-yearly":
            quarter = "Half-year"
        elif period.casefold() == "annual" or relating_to.casefold() == "annual":
            quarter = "Annual"

        detail_url = str(row.get("resultDetailedDataLink") or "").strip()
        xbrl_url = str(row.get("xbrl") or "").strip()
        source_url = detail_url if detail_url not in {"", "-"} else xbrl_url
        if source_url in {"", "-"} or source_url.endswith("/-"):
            source_url = ""
        if source_url and not official_url(source_url):
            raise ExchangeSourceError("NSE returned a financial-results link outside its approved official hosts.")

        labels = [value for value in (relating_to, financial_year) if value]
        for field in ("audited", "consolidated", "indAs"):
            value = str(row.get(field) or "").strip()
            if value and value != "-":
                labels.append(value)
        title = "Financial Results"
        if labels:
            title += " - " + ", ".join(labels)
        return FilingRecord(
            exchange="NSE",
            company=str(row.get("companyName") or "").strip(),
            ticker=str(row.get("symbol") or symbol).strip().upper(),
            title=title,
            original_category="Financial Results",
            publication_date=_date_text(row.get("filingDate") or row.get("broadCastDate")),
            source_url=source_url,
            reporting_period="; ".join(value for value in (financial_year, relating_to, period) if value),
            fiscal_year=fiscal_year,
            quarter=quarter,
            file_format=_format_from_url(source_url),
            description="; ".join(value for value in (str(row.get("consolidated") or "").strip(), str(row.get("audited") or "").strip()) if value),
            document_id=str(row.get("seqNumber") or ""),
        )


class BSEAdapter(_RateLimitedAdapter):
    exchange = "BSE"
    search_url = BSE_SEARCH_URL

    def search(
        self,
        scrip_code: str,
        from_date: date,
        to_date: date,
        limit: int = 100,
        offset: int = 0,
        subject: str = "",
    ) -> tuple[list[FilingRecord], int]:
        scrip_code = scrip_code.strip()
        if not re.fullmatch(r"\d{1,10}", scrip_code):
            raise ValueError("Enter an exact numeric BSE scrip code.")
        if from_date > to_date:
            raise ValueError("The publication start date must be on or before the end date.")
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("Result pages must be 1-100 records with a non-negative offset.")

        params = {
            "pageno": offset // limit + 1,
            "strCat": "-1",
            "strPrevDate": from_date.strftime("%Y%m%d"),
            "strScrip": scrip_code,
            "strSearch": "P",
            "strToDate": to_date.strftime("%Y%m%d"),
            "strType": "C",
            "subcategory": "-1",
        }
        response = self._get(BSE_API_URL, params)
        try:
            payload = response.json()
        except (requests.JSONDecodeError, ValueError) as error:
            raise ExchangeSourceError("BSE returned an unexpected non-JSON filing response.") from error

        rows = payload if isinstance(payload, list) else next(
            (value for value in payload.values() if isinstance(value, list)), []
        ) if isinstance(payload, dict) else []
        if not rows and payload:
            raise ExchangeSourceError("BSE returned an unrecognized filing response shape.")
        records = [self._record(row, scrip_code) for row in rows if isinstance(row, dict)]
        if subject.strip():
            term = subject.casefold().strip()
            records = [record for record in records if term in f"{record.title} {record.description}".casefold()]
        total = min(len(records), MAX_SEARCH_RESULTS)
        start = offset % limit
        return records[:MAX_SEARCH_RESULTS][start : start + limit], total

    @staticmethod
    def _record(row: dict[str, Any], scrip_code: str) -> FilingRecord:
        lower = {str(key).casefold(): value for key, value in row.items()}

        def first(*keys: str) -> str:
            return next((str(lower[key.casefold()]).strip() for key in keys if lower.get(key.casefold())), "")

        title = first("NEWSSUB", "SUBJECT", "HEADLINE", "TITLE")
        source_url = first("ATTACHMENTNAME", "ATTACHMENT", "URL", "SOURCEURL")
        if source_url and not urlparse(source_url).scheme:
            source_url = urljoin("https://www.bseindia.com/", source_url.lstrip("/"))
        if source_url and not official_url(source_url):
            raise ExchangeSourceError("BSE returned a filing link outside its approved official hosts.")
        if not title:
            raise ExchangeSourceError("BSE filing response did not contain a recognized title field.")
        category = first("CATEGORYNAME", "CATEGORY", "SUBCATEGORYNAME", "SUBCATEGORY") or "Other company filings"
        company = first("SLONGNAME", "COMPANY", "COMPANYNAME", "SCRIPNAME")
        fiscal_year, quarter, reporting_period = _period_fields(title)
        return FilingRecord(
            exchange="BSE",
            company=company,
            ticker=first("SCRIP_CD", "SCRIPCODE", "SCRIP_CODE") or scrip_code,
            title=title,
            original_category=category,
            publication_date=_date_text(first("NEWS_DT", "DATE", "PUBLISHDATE")),
            source_url=source_url,
            reporting_period=reporting_period,
            fiscal_year=fiscal_year,
            quarter=quarter,
            file_format=_format_from_url(source_url),
            file_size=first("FILESIZE", "ATTACHMENTSIZE"),
            description=first("NEWS", "DESCRIPTION", "DETAILS"),
            document_id=first("NEWSID", "ID"),
        )


def official_search_link(exchange: str, query: str = "") -> str:
    del query
    return BSE_SEARCH_URL if exchange.upper() == "BSE" else NSE_SEARCH_URL


def save_search_cache(cache_file: Path, records: list[FilingRecord]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps([record.as_dict() for record in records], ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def load_search_cache(cache_file: Path) -> list[FilingRecord]:
    if not cache_file.exists():
        return []
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return []
        return [FilingRecord(**item) for item in payload if isinstance(item, dict)]
    except (OSError, json.JSONDecodeError, TypeError):
        return []


def load_document_catalog(catalog_file: Path) -> dict[str, dict[str, Any]]:
    if not catalog_file.exists():
        return {}
    try:
        payload = json.loads(catalog_file.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_document_catalog(
    catalog_file: Path,
    catalog: dict[str, dict[str, Any]],
) -> None:
    catalog_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = catalog_file.with_suffix(catalog_file.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(catalog, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(catalog_file)