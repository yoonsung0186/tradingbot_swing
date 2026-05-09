from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .alpaca import AlpacaError, AlpacaMarketDataClient, AlpacaNewsItem, alpaca_credentials_available
from .official_research import Filing, OfficialResearch, SecClient
from .realtime_session import classify_research_signals


NEW_YORK = ZoneInfo("America/New_York")


SYMBOL_INDUSTRY = {
    "AAPL": "technology",
    "MSFT": "technology",
    "NVDA": "semiconductors",
    "AVGO": "semiconductors",
    "AMD": "semiconductors",
    "INTC": "semiconductors",
    "QCOM": "semiconductors",
    "GOOGL": "communication",
    "META": "communication",
    "NFLX": "communication",
    "AMZN": "consumer_internet",
    "TSLA": "consumer_discretionary",
    "COST": "consumer_defensive",
    "WMT": "consumer_defensive",
    "HD": "consumer_discretionary",
    "NKE": "consumer_discretionary",
    "MCD": "consumer_defensive",
    "ORCL": "software",
    "CRM": "software",
    "ADBE": "software",
    "JPM": "financials",
    "BAC": "financials",
    "GS": "financials",
    "MS": "financials",
    "V": "payments",
    "MA": "payments",
    "AXP": "payments",
    "LLY": "healthcare",
    "UNH": "healthcare",
    "JNJ": "healthcare",
    "ABBV": "healthcare",
    "MRK": "healthcare",
    "XOM": "energy",
    "CVX": "energy",
    "COP": "energy",
    "GE": "industrials",
    "CAT": "industrials",
    "BA": "industrials",
    "NOW": "software",
    "PANW": "software",
    "CRWD": "software",
    "PLTR": "software",
    "SNOW": "software",
    "ARM": "semiconductors",
    "MU": "semiconductors",
    "KLAC": "semiconductors",
    "LRCX": "semiconductors",
    "AMAT": "semiconductors",
    "ASML": "semiconductors",
    "TSM": "semiconductors",
    "MRVL": "semiconductors",
    "APP": "software",
    "UBER": "consumer_internet",
    "ABNB": "consumer_internet",
    "SHOP": "consumer_internet",
    "COIN": "financials",
    "MSTR": "software",
    "HOOD": "financials",
    "SMCI": "semiconductors",
    "DELL": "technology",
    "ANET": "technology",
    "CEG": "utilities",
    "VRT": "industrials",
    "ETN": "industrials",
    "PWR": "industrials",
    "NVO": "healthcare",
    "ISRG": "healthcare",
    "REGN": "healthcare",
    "TMO": "healthcare",
    "LIN": "materials",
    "LMT": "industrials",
    "RTX": "industrials",
    "HON": "industrials",
    "URI": "industrials",
    "DE": "industrials",
    "FSLR": "energy",
    "ENPH": "energy",
    "RCL": "consumer_discretionary",
    "BKNG": "consumer_discretionary",
    "MAR": "consumer_discretionary",
    "TQQQ": "technology",
    "SOXL": "semiconductors",
    "TECL": "technology",
    "UPRO": "industrials",
    "SPXL": "industrials",
    "FNGU": "communication",
    "BULZ": "technology",
    "TNA": "financials",
    "LABU": "healthcare",
}

EXCLUDED_NEWS_MARKERS = (
    "reddit",
    "stocktwits",
    "youtube",
    "message board",
    "blogspot",
    "seekingalpha.com/instablog",
)

POSITIVE_TERMS = (
    "raises guidance",
    "raised guidance",
    "beats estimates",
    "beat estimates",
    "record revenue",
    "profit tops",
    "buyback",
    "repurchase",
    "dividend increase",
    "fda approval",
    "wins approval",
    "major contract",
    "partnership",
    "acquisition",
    "expands production",
    "strong demand",
)

RISK_TERMS = (
    "sec investigation",
    "accounting probe",
    "restatement",
    "going concern",
    "bankruptcy",
    "chapter 11",
    "downgrade",
    "cuts guidance",
    "cut guidance",
    "misses estimates",
    "offering",
    "secondary offering",
    "dilution",
    "recall",
    "lawsuit",
    "antitrust",
)

EVENT_FORMS = {"8-K", "6-K"}
OFFERING_FORMS = {"S-1", "S-3", "424B5", "FWP"}


@dataclass(frozen=True)
class SwingResearchScore:
    symbol: str
    as_of: date
    allow_entry: bool
    risk_level: str
    risk_score: int
    positive_score: int
    industry_score: float
    score_adjustment: float
    position_multiplier: float
    reasons: tuple[str, ...]
    news_count: int
    filings_count: int


class SwingResearchProvider:
    def __init__(
        self,
        symbols: list[str],
        start: date,
        end: date,
        source: str = "sec-alpaca",
        news_lookback_days: int = 7,
        filing_lookback_days: int = 14,
        industry_lookback_days: int = 14,
        block_risk_score: int = 4,
        caution_risk_score: int = 2,
        include_filing_text: bool = True,
        max_news_pages_per_chunk: int = 8,
        cache_dir: Path = Path("data/sec_cache"),
    ) -> None:
        self.symbols = _dedupe(symbols)
        self.start = start
        self.end = end
        self.source = source
        self.news_lookback_days = news_lookback_days
        self.filing_lookback_days = filing_lookback_days
        self.industry_lookback_days = industry_lookback_days
        self.block_risk_score = block_risk_score
        self.caution_risk_score = caution_risk_score
        self.include_filing_text = include_filing_text
        self.max_news_pages_per_chunk = max_news_pages_per_chunk
        self.sec_client = SecClient(cache_dir=cache_dir)
        self.news_by_symbol: dict[str, list[AlpacaNewsItem]] = {symbol: [] for symbol in self.symbols}
        self.filings_by_symbol: dict[str, list[Filing]] = {symbol: [] for symbol in self.symbols}
        self.company_by_symbol: dict[str, tuple[str, str]] = {}
        self.errors: list[str] = []
        self._score_cache: dict[tuple[str, date], SwingResearchScore] = {}
        self._filing_text_cache: dict[str, str] = {}
        self._checked = 0
        self._blocked = 0
        self._caution = 0
        self._positive = 0
        self._load_sec()
        if source == "sec-alpaca":
            self._load_alpaca_news()

    def score(self, symbol: str, as_of: date) -> SwingResearchScore:
        key = (symbol.upper(), as_of)
        if key in self._score_cache:
            return self._score_cache[key]

        upper = symbol.upper()
        now = datetime.combine(as_of, time(23, 59, 59), tzinfo=NEW_YORK).astimezone(timezone.utc)
        news = self._news_as_of(upper, now, self.news_lookback_days)
        filings = self._filings_as_of(upper, as_of, self.filing_lookback_days)
        cik, company_name = self.company_by_symbol.get(upper, ("", upper))
        official = OfficialResearch(
            symbol=upper,
            cik=cik,
            company_name=company_name,
            latest_filings=filings,
            facts={},
            official_queries=[],
        )
        official_texts = self._official_texts(filings[:2]) if self.include_filing_text else []
        snapshot = classify_research_signals(
            upper,
            news,
            official if cik or filings else None,
            now,
            official_texts=official_texts,
            block_risk_score=self.block_risk_score,
            caution_risk_score=self.caution_risk_score,
        )
        industry_score = self._industry_score(upper, now)
        score_adjustment = (
            min(snapshot.positive_score, 4) * 0.025
            + industry_score * 0.020
            - min(snapshot.risk_score, 6) * 0.040
        )
        if snapshot.risk_level == "blocked":
            position_multiplier = 0.0
        elif snapshot.risk_level == "caution":
            position_multiplier = 0.50
        else:
            position_multiplier = 1.0 + min(snapshot.positive_score, 3) * 0.05 + max(industry_score, 0.0) * 0.03
            position_multiplier = min(position_multiplier, 1.20)

        reasons = list(snapshot.reasons)
        if industry_score:
            reasons.append(f"industry_score={industry_score:.2f}")
        if not reasons:
            reasons.append("research clear")
        result = SwingResearchScore(
            symbol=upper,
            as_of=as_of,
            allow_entry=snapshot.allow_entry,
            risk_level=snapshot.risk_level,
            risk_score=snapshot.risk_score,
            positive_score=snapshot.positive_score,
            industry_score=industry_score,
            score_adjustment=score_adjustment,
            position_multiplier=position_multiplier,
            reasons=tuple(reasons[:8]),
            news_count=len(news),
            filings_count=len(filings),
        )
        self._score_cache[key] = result
        self._checked += 1
        if result.risk_level == "blocked":
            self._blocked += 1
        elif result.risk_level == "caution":
            self._caution += 1
        if result.positive_score > 0 or result.industry_score > 0:
            self._positive += 1
        return result

    def summary(self) -> dict:
        return {
            "source": self.source,
            "symbols": len(self.symbols),
            "news_items": sum(len(items) for items in self.news_by_symbol.values()),
            "filings": sum(len(items) for items in self.filings_by_symbol.values()),
            "checked_signals": self._checked,
            "blocked_signals": self._blocked,
            "caution_signals": self._caution,
            "positive_signals": self._positive,
            "errors": self.errors[:10],
            "news_lookback_days": self.news_lookback_days,
            "filing_lookback_days": self.filing_lookback_days,
            "industry_lookback_days": self.industry_lookback_days,
        }

    def _load_sec(self) -> None:
        for symbol in self.symbols:
            try:
                research = self.sec_client.research(symbol, limit=250, include_facts=False)
            except Exception as exc:
                self.errors.append(f"{symbol}: SEC unavailable: {exc}")
                continue
            self.company_by_symbol[symbol] = (research.cik, research.company_name)
            start = self.start - timedelta(days=self.filing_lookback_days)
            self.filings_by_symbol[symbol] = [
                filing
                for filing in research.latest_filings
                if start <= _parse_date(filing.filed) <= self.end
            ]

    def _load_alpaca_news(self) -> None:
        if not alpaca_credentials_available():
            self.errors.append("Alpaca credentials unavailable; SEC-only research used")
            return
        client = AlpacaMarketDataClient()
        start_dt = datetime.combine(self.start - timedelta(days=self.news_lookback_days), time.min, tzinfo=NEW_YORK)
        end_dt = datetime.combine(self.end, time.max, tzinfo=NEW_YORK)
        chunk_start = start_dt
        seen_ids: set[str] = set()
        while chunk_start < end_dt:
            chunk_end = min(chunk_start + timedelta(days=30), end_dt)
            try:
                news_items = client.historical_news(
                    self.symbols,
                    chunk_start,
                    chunk_end,
                    max_pages=self.max_news_pages_per_chunk,
                )
            except AlpacaError as exc:
                self.errors.append(f"Alpaca news unavailable: {exc}")
                return
            for item in news_items:
                if _is_excluded_news_item(item):
                    continue
                news_id = str(item.id)
                if news_id in seen_ids:
                    continue
                seen_ids.add(news_id)
                for item_symbol in item.symbols:
                    if item_symbol in self.news_by_symbol:
                        self.news_by_symbol[item_symbol].append(item)
            chunk_start = chunk_end + timedelta(seconds=1)
        for symbol in self.news_by_symbol:
            self.news_by_symbol[symbol].sort(key=lambda item: item.created_at)

    def _news_as_of(self, symbol: str, as_of: datetime, lookback_days: int) -> list[AlpacaNewsItem]:
        start = as_of - timedelta(days=lookback_days)
        return [
            item
            for item in self.news_by_symbol.get(symbol, [])
            if start <= item.created_at.astimezone(timezone.utc) <= as_of
        ]

    def _filings_as_of(self, symbol: str, as_of: date, lookback_days: int) -> list[Filing]:
        start = as_of - timedelta(days=lookback_days)
        return [
            filing
            for filing in self.filings_by_symbol.get(symbol, [])
            if start <= _parse_date(filing.filed) <= as_of
        ]

    def _official_texts(self, filings: list[Filing]) -> list[str]:
        texts: list[str] = []
        for filing in filings:
            if filing.form not in (EVENT_FORMS | OFFERING_FORMS):
                continue
            key = filing.accession
            if key not in self._filing_text_cache:
                try:
                    self._filing_text_cache[key] = self.sec_client.filing_text(filing, max_chars=30_000)
                except Exception:
                    self._filing_text_cache[key] = ""
            if self._filing_text_cache[key]:
                texts.append(self._filing_text_cache[key])
        return texts

    def _industry_score(self, symbol: str, as_of: datetime) -> float:
        industry = SYMBOL_INDUSTRY.get(symbol)
        if not industry:
            return 0.0
        start = as_of - timedelta(days=self.industry_lookback_days)
        positive = 0
        risk = 0
        for peer, peer_industry in SYMBOL_INDUSTRY.items():
            if peer_industry != industry:
                continue
            for item in self.news_by_symbol.get(peer, []):
                item_time = item.created_at.astimezone(timezone.utc)
                if not (start <= item_time <= as_of):
                    continue
                text = f"{item.headline} {item.summary}".lower()
                if any(term in text for term in POSITIVE_TERMS):
                    positive += 1
                if any(term in text for term in RISK_TERMS):
                    risk += 1
        if positive == 0 and risk == 0:
            return 0.0
        return max(min((positive - risk) / 5, 3.0), -3.0)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return date.min


def _is_excluded_news_item(item: AlpacaNewsItem) -> bool:
    text = f"{item.source} {item.url} {item.headline}".lower()
    return any(marker in text for marker in EXCLUDED_NEWS_MARKERS)


def _dedupe(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for symbol in symbols:
        upper = symbol.upper()
        if upper and upper not in seen:
            seen.add(upper)
            result.append(upper)
    return result
