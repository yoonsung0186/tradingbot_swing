from __future__ import annotations

import html
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{document}"


@dataclass(frozen=True)
class Filing:
    symbol: str
    form: str
    filed: str
    accession: str
    document: str
    url: str


@dataclass(frozen=True)
class OfficialResearch:
    symbol: str
    cik: str
    company_name: str
    latest_filings: list[Filing]
    facts: dict[str, float | str]
    official_queries: list[str]


class SecClient:
    def __init__(
        self,
        cache_dir: Path = Path("data/sec_cache"),
        timeout: int = 20,
        cache_ttl_seconds: int = 6 * 60 * 60,
    ) -> None:
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.cache_ttl_seconds = cache_ttl_seconds
        self.user_agent = os.getenv("SEC_USER_AGENT", "us-ai-trading-agent contact@example.com")

    def research(self, symbol: str, limit: int = 5, include_facts: bool = True) -> OfficialResearch:
        mapping = self._ticker_mapping()
        key = symbol.upper()
        if key not in mapping:
            raise ValueError(f"No SEC CIK mapping found for {symbol}")
        cik, company_name = mapping[key]
        submissions = self._json(SEC_SUBMISSIONS_URL.format(cik=cik), self.cache_dir / f"{cik}_submissions.json")
        facts = self._company_facts(cik) if include_facts else {}
        return OfficialResearch(
            symbol=key,
            cik=cik,
            company_name=company_name,
            latest_filings=self._latest_filings(key, cik, submissions, limit=limit),
            facts=facts,
            official_queries=official_google_queries(key, company_name),
        )

    def filing_text(self, filing: Filing, max_chars: int = 60_000) -> str:
        filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{filing.accession}_{filing.document}.txt")
        cache_path = self.cache_dir / "filings" / filename
        if cache_path.exists():
            age_seconds = datetime.now().timestamp() - cache_path.stat().st_mtime
            if age_seconds < self.cache_ttl_seconds:
                return cache_path.read_text(encoding="utf-8", errors="replace")[:max_chars]
        request = urllib.request.Request(
            filing.url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,text/plain",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read(max_chars * 4).decode("utf-8", errors="replace")
        text = _html_to_text(raw)[:max_chars]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
        return text

    def _ticker_mapping(self) -> dict[str, tuple[str, str]]:
        payload = self._json(SEC_TICKERS_URL, self.cache_dir / "company_tickers.json")
        result: dict[str, tuple[str, str]] = {}
        for item in payload.values():
            ticker = item["ticker"].upper()
            cik = str(item["cik_str"]).zfill(10)
            result[ticker] = (cik, item["title"])
        return result

    def _company_facts(self, cik: str) -> dict[str, float | str]:
        try:
            payload = self._json(SEC_FACTS_URL.format(cik=cik), self.cache_dir / f"{cik}_facts.json")
        except Exception:
            return {}
        facts = payload.get("facts", {}).get("us-gaap", {})
        return {
            "Revenue": _latest_fact(facts, ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"]),
            "NetIncomeLoss": _latest_fact(facts, ["NetIncomeLoss"]),
            "Assets": _latest_fact(facts, ["Assets"]),
            "Liabilities": _latest_fact(facts, ["Liabilities"]),
            "EPSDiluted": _latest_fact(facts, ["EarningsPerShareDiluted"]),
        }

    def _latest_filings(self, symbol: str, cik: str, submissions: dict, limit: int) -> list[Filing]:
        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form") or []
        dates = recent.get("filingDate") or []
        accessions = recent.get("accessionNumber") or []
        documents = recent.get("primaryDocument") or []
        accepted_forms = {"10-K", "10-Q", "8-K", "20-F", "6-K", "S-1", "S-3", "424B5", "FWP"}
        filings: list[Filing] = []
        for form, filed, accession, document in zip(forms, dates, accessions, documents):
            if form not in accepted_forms:
                continue
            accession_path = accession.replace("-", "")
            cik_int = str(int(cik))
            url = SEC_ARCHIVES_URL.format(cik_int=cik_int, accession=accession_path, document=document)
            filings.append(
                Filing(
                    symbol=symbol,
                    form=form,
                    filed=filed,
                    accession=accession,
                    document=document,
                    url=url,
                )
            )
            if len(filings) >= limit:
                break
        return filings

    def _json(self, url: str, cache_path: Path) -> dict:
        if cache_path.exists():
            age_seconds = datetime.now().timestamp() - cache_path.stat().st_mtime
            if age_seconds < self.cache_ttl_seconds:
                return json.loads(cache_path.read_text(encoding="utf-8"))
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            text = response.read().decode("utf-8")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
        return json.loads(text)


def official_google_queries(symbol: str, company_name: str) -> list[str]:
    queries = [
        f"{symbol} latest 10-Q site:sec.gov",
        f"{symbol} latest 10-K site:sec.gov",
        f"{company_name} investor relations official",
        f"{company_name} earnings release official",
    ]
    return [f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}" for query in queries]


def format_research(items: list[OfficialResearch]) -> str:
    lines: list[str] = []
    for item in items:
        lines.append(f"{item.symbol} / {item.company_name} / CIK {item.cik}")
        if item.facts:
            fact_parts = []
            for key, value in item.facts.items():
                if value not in {None, ""}:
                    fact_parts.append(f"{key}={value}")
            if fact_parts:
                lines.append("  Facts: " + "; ".join(fact_parts))
        lines.append("  Latest official filings:")
        for filing in item.latest_filings:
            lines.append(f"  - {filing.filed} {filing.form}: {filing.url}")
        lines.append("  Official Google queries:")
        for query in item.official_queries:
            lines.append(f"  - {query}")
        lines.append("")
    return "\n".join(lines).strip()


def _latest_fact(facts: dict, names: list[str]) -> float | str:
    for name in names:
        units = facts.get(name, {}).get("units", {})
        for unit_values in units.values():
            values = [item for item in unit_values if "val" in item and item.get("filed")]
            if values:
                values.sort(key=lambda item: item.get("filed", ""))
                latest = values[-1]
                return latest["val"]
    return ""


def _html_to_text(raw: str) -> str:
    text = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()
