from __future__ import annotations

import csv
import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from time import sleep
from zoneinfo import ZoneInfo

from .alpaca import AlpacaError, AlpacaMarketDataClient, AlpacaNewsItem, AlpacaPaperClient, alpaca_credentials_available
from .data import DataError, YahooChartClient
from .models import IntradayBar
from .official_research import OfficialResearch, SecClient
from .realtime import PriceTick, RealtimeConfig, RealtimeDecision, RealtimeReactiveModel


KST = ZoneInfo("Asia/Seoul")
NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class RealtimeSessionConfig:
    symbols: list[str]
    cash: float
    provider: str = "alpaca"
    feed: str = "iex"
    poll_seconds: int = 10
    until: datetime | None = None
    risk_mode: str = "stable"
    output_root: Path = Path("reports")
    wait_for_credentials: bool = False
    credential_check_seconds: int = 60
    dynamic_universe: bool = False
    scan_symbols: list[str] | None = None
    scan_interval_seconds: int = 180
    scan_max_symbols: int = 60
    dynamic_max_symbols: int = 16
    top_surging_symbols: int = 6
    min_volume_ratio: float = 1.6
    min_recent_dollar_volume: float = 3_000_000.0
    min_short_return_pct: float = 0.0015
    market_filter: bool = True
    market_symbols: tuple[str, ...] = ("SPY", "QQQ")
    market_lookback_ticks: int = 6
    min_market_momentum_pct: float = -0.003
    research_filter: bool = True
    research_interval_seconds: int = 300
    research_max_symbols: int = 20
    research_news_lookback_minutes: int = 180
    research_block_risk_score: int = 4
    research_caution_risk_score: int = 2
    sec_cache_ttl_seconds: int = 300


@dataclass(frozen=True)
class SurgeScanConfig:
    recent_bars: int = 5
    baseline_bars: int = 45
    min_volume_ratio: float = 1.6
    min_recent_dollar_volume: float = 3_000_000.0
    min_short_return_pct: float = 0.0015
    max_candidates: int = 6


@dataclass(frozen=True)
class SurgeCandidate:
    symbol: str
    score: float
    price: float
    volume_ratio: float
    recent_dollar_volume: float
    short_return: float
    day_return: float
    bars: int


@dataclass(frozen=True)
class MarketSnapshot:
    risk_on: bool
    reason: str
    momentums: dict[str, float]


@dataclass(frozen=True)
class MarketHoursSnapshot:
    is_open: bool
    reason: str
    open_at: datetime | None = None
    close_at: datetime | None = None
    source: str = "fallback"


@dataclass(frozen=True)
class ResearchSymbolSnapshot:
    symbol: str
    checked_at: datetime
    allow_entry: bool
    risk_level: str
    risk_score: int
    positive_score: int
    reasons: tuple[str, ...]
    news_count: int
    filings_count: int
    latest_news_headline: str = ""
    latest_news_source: str = ""
    latest_news_url: str = ""
    latest_filing_form: str = ""
    latest_filing_date: str = ""
    latest_filing_url: str = ""


@dataclass(frozen=True)
class ResearchSnapshot:
    enabled: bool
    checked_at: datetime
    reason: str
    by_symbol: dict[str, ResearchSymbolSnapshot]

    def symbol_snapshot(self, symbol: str) -> ResearchSymbolSnapshot | None:
        return self.by_symbol.get(symbol.upper())

    def allows_entry(self, symbol: str) -> bool:
        item = self.symbol_snapshot(symbol)
        return True if item is None else item.allow_entry


@dataclass
class AdaptiveEvent:
    timestamp: datetime
    event: str
    detail: str
    entry_momentum_pct: float
    max_position_pct: float
    max_spread_pct: float
    cooldown_ticks: int


class AdaptiveRiskController:
    def __init__(self, base_config: RealtimeConfig, max_session_loss_pct: float = 0.03) -> None:
        self.base_config = base_config
        self.peak_equity = 0.0
        self.loss_streak = 0
        self.win_streak = 0
        self.tightening_steps = 0
        self.cash_guard = False
        self.max_session_loss_pct = max_session_loss_pct

    def observe_equity(
        self,
        model: RealtimeReactiveModel,
        timestamp: datetime,
        start_equity: float,
        equity: float,
    ) -> AdaptiveEvent | None:
        self.peak_equity = max(self.peak_equity, equity)
        peak_drawdown = equity / self.peak_equity - 1 if self.peak_equity else 0.0
        session_return = equity / start_equity - 1 if start_equity else 0.0
        if self.cash_guard:
            return None
        if session_return <= -self.max_session_loss_pct or peak_drawdown <= -self.max_session_loss_pct:
            updated = replace(model.config, max_position_pct=0.0, cooldown_ticks=max(model.config.cooldown_ticks, 60))
            model.update_config(updated)
            self.cash_guard = True
            return self._event(
                timestamp,
                "CASH_GUARD",
                "손실 한도에 도달해 신규 진입을 중단했습니다.",
                model.config,
            )
        return None

    def observe_closed_trade(
        self,
        model: RealtimeReactiveModel,
        timestamp: datetime,
        pnl_pct: float,
    ) -> AdaptiveEvent | None:
        if self.cash_guard:
            return None
        if pnl_pct < 0:
            self.loss_streak += 1
            self.win_streak = 0
        else:
            self.win_streak += 1
            self.loss_streak = 0

        if self.loss_streak >= 2:
            current = model.config
            updated = replace(
                current,
                entry_momentum_pct=min(current.entry_momentum_pct * 1.35, 0.012),
                max_position_pct=max(current.max_position_pct * 0.55, 0.01),
                max_spread_pct=max(current.max_spread_pct * 0.75, 0.0008),
                cooldown_ticks=min(current.cooldown_ticks + 8, 80),
            )
            model.update_config(updated)
            self.loss_streak = 0
            self.tightening_steps += 1
            return self._event(
                timestamp,
                "MODEL_TIGHTENED",
                "연속 손실이 발생해 진입 기준을 강화하고 포지션 비중을 낮췄습니다.",
                model.config,
            )

        if self.win_streak >= 3 and self.tightening_steps > 0:
            current = model.config
            updated = replace(
                current,
                entry_momentum_pct=max(current.entry_momentum_pct * 0.92, self.base_config.entry_momentum_pct),
                max_position_pct=min(current.max_position_pct * 1.15, self.base_config.max_position_pct),
                max_spread_pct=min(current.max_spread_pct * 1.08, self.base_config.max_spread_pct),
                cooldown_ticks=max(current.cooldown_ticks - 4, self.base_config.cooldown_ticks),
            )
            model.update_config(updated)
            self.win_streak = 0
            self.tightening_steps = max(self.tightening_steps - 1, 0)
            return self._event(
                timestamp,
                "MODEL_RELAXED",
                "연속 이익 후 기준을 초기 설정 쪽으로 일부 되돌렸습니다.",
                model.config,
            )
        return None

    @staticmethod
    def _event(timestamp: datetime, event: str, detail: str, config: RealtimeConfig) -> AdaptiveEvent:
        return AdaptiveEvent(
            timestamp=timestamp,
            event=event,
            detail=detail,
            entry_momentum_pct=config.entry_momentum_pct,
            max_position_pct=config.max_position_pct,
            max_spread_pct=config.max_spread_pct,
            cooldown_ticks=config.cooldown_ticks,
        )


class MarketFilter:
    def __init__(self, symbols: tuple[str, ...], lookback_ticks: int, min_momentum_pct: float) -> None:
        self.symbols = tuple(symbol.upper() for symbol in symbols)
        self.windows: dict[str, deque[PriceTick]] = defaultdict(lambda: deque(maxlen=max(lookback_ticks, 2)))
        self.min_momentum_pct = min_momentum_pct

    def update(self, tick: PriceTick) -> None:
        if tick.symbol.upper() in self.symbols:
            self.windows[tick.symbol.upper()].append(tick)

    def snapshot(self) -> MarketSnapshot:
        momentums: dict[str, float] = {}
        missing: list[str] = []
        for symbol in self.symbols:
            window = self.windows[symbol]
            if len(window) < 2 or window[0].price <= 0:
                missing.append(symbol)
                continue
            momentums[symbol] = window[-1].price / window[0].price - 1

        if not momentums:
            return MarketSnapshot(True, "market filter waiting for enough SPY/QQQ ticks", {})

        weak = {symbol: value for symbol, value in momentums.items() if value <= self.min_momentum_pct}
        if weak:
            formatted = ", ".join(f"{symbol} {value:.2%}" for symbol, value in weak.items())
            return MarketSnapshot(False, f"market momentum weak: {formatted}", momentums)
        if missing:
            return MarketSnapshot(True, f"market filter partial data; missing {', '.join(missing)}", momentums)
        return MarketSnapshot(True, "market momentum acceptable", momentums)


class AlpacaMarketHoursCalendar:
    def __init__(self, provider: str) -> None:
        self.enabled = provider.lower() == "alpaca"
        self.client: AlpacaPaperClient | None = None
        self.cache: dict[str, dict | None] = {}

    def snapshot(self, timestamp: datetime) -> MarketHoursSnapshot:
        if not self.enabled:
            return self._fallback(timestamp, "calendar unavailable for non-Alpaca provider")
        ny_time = timestamp.astimezone(NEW_YORK)
        key = ny_time.date().isoformat()
        try:
            if self.client is None:
                self.client = AlpacaPaperClient()
            if key not in self.cache:
                items = self.client.calendar(start=key, end=key)
                self.cache[key] = items[0] if items else None
            item = self.cache[key]
            if not item:
                return MarketHoursSnapshot(False, f"Alpaca calendar closed on {key}", source="alpaca_calendar")
            open_at = _parse_alpaca_calendar_datetime(item, key, "open")
            close_at = _parse_alpaca_calendar_datetime(item, key, "close")
            if open_at is None or close_at is None:
                return self._fallback(timestamp, "Alpaca calendar missing open/close; fallback regular hours")
            if open_at <= ny_time < close_at:
                return MarketHoursSnapshot(
                    True,
                    f"Alpaca calendar open until {close_at.strftime('%H:%M %Z')}",
                    open_at=open_at,
                    close_at=close_at,
                    source="alpaca_calendar",
                )
            if ny_time < open_at:
                reason = f"Alpaca calendar waiting for open {open_at.strftime('%H:%M %Z')}"
            else:
                reason = f"Alpaca calendar closed after {close_at.strftime('%H:%M %Z')}"
            return MarketHoursSnapshot(False, reason, open_at=open_at, close_at=close_at, source="alpaca_calendar")
        except Exception as exc:
            return self._fallback(timestamp, f"Alpaca calendar unavailable; fallback regular hours: {exc}")

    @staticmethod
    def _fallback(timestamp: datetime, reason: str) -> MarketHoursSnapshot:
        return MarketHoursSnapshot(_is_us_regular_market(timestamp), reason, source="fallback_regular_hours")


class ResearchRiskMonitor:
    def __init__(
        self,
        provider: str,
        feed: str,
        news_lookback_minutes: int,
        block_risk_score: int,
        caution_risk_score: int,
        sec_cache_ttl_seconds: int,
    ) -> None:
        self.provider = provider.lower()
        self.feed = feed
        self.news_lookback_minutes = news_lookback_minutes
        self.block_risk_score = block_risk_score
        self.caution_risk_score = caution_risk_score
        self.sec_client = SecClient(cache_ttl_seconds=sec_cache_ttl_seconds)
        self.alpaca_client: AlpacaMarketDataClient | None = None
        if self.provider == "alpaca":
            self.alpaca_client = AlpacaMarketDataClient()

    def refresh(self, symbols: list[str], now: datetime) -> ResearchSnapshot:
        requested = _dedupe_symbols(symbols)
        if not requested:
            return ResearchSnapshot(True, now, "research filter enabled but no symbols", {})

        reasons: list[str] = []
        news_by_symbol: dict[str, list[AlpacaNewsItem]] = defaultdict(list)
        if self.alpaca_client is not None:
            try:
                news_items = self.alpaca_client.latest_news(
                    requested,
                    lookback_minutes=self.news_lookback_minutes,
                    limit=50,
                    include_content=False,
                )
                requested_set = set(requested)
                for item in news_items:
                    item_symbols = set(item.symbols) & requested_set
                    for symbol in item_symbols:
                        if not _is_excluded_news_item(item):
                            news_by_symbol[symbol].append(item)
            except Exception as exc:
                reasons.append(f"news unavailable: {exc}")
        else:
            reasons.append("news unavailable: provider is not alpaca")

        by_symbol: dict[str, ResearchSymbolSnapshot] = {}
        for symbol in requested:
            official: OfficialResearch | None = None
            sec_error = ""
            official_texts: list[str] = []
            try:
                official = self.sec_client.research(symbol, limit=3, include_facts=False)
                for filing in official.latest_filings[:2]:
                    if filing.form in (_EVENT_FORMS | _OFFERING_FORMS):
                        try:
                            official_texts.append(self.sec_client.filing_text(filing, max_chars=40_000))
                        except Exception:
                            continue
            except Exception as exc:
                sec_error = f"SEC unavailable: {exc}"
            item = classify_research_signals(
                symbol,
                news_by_symbol.get(symbol, []),
                official,
                now,
                official_texts=official_texts,
                block_risk_score=self.block_risk_score,
                caution_risk_score=self.caution_risk_score,
                extra_reason=sec_error,
            )
            by_symbol[symbol] = item

        blocked = [symbol for symbol, item in by_symbol.items() if item.risk_level == "blocked"]
        caution = [symbol for symbol, item in by_symbol.items() if item.risk_level == "caution"]
        status_parts = []
        if blocked:
            status_parts.append(f"blocked={','.join(blocked)}")
        if caution:
            status_parts.append(f"caution={','.join(caution)}")
        if reasons:
            status_parts.extend(reasons)
        reason = "; ".join(status_parts) if status_parts else "official/news filter clear"
        return ResearchSnapshot(True, now, reason, by_symbol)


_EXCLUDED_NEWS_MARKERS = (
    "reddit",
    "stocktwits",
    "youtube",
    "message board",
    "forum",
    "opinion",
    "commentary",
    "rumor",
    "rumour",
)

_RISK_RULES: tuple[tuple[tuple[str, ...], int, str], ...] = (
    (("bankruptcy", "chapter 11", "insolvency", "going concern"), 4, "bankruptcy/going-concern risk"),
    (("delisting", "nasdaq notice", "nyse notice", "trading halt", "halted"), 4, "listing or trading-halt risk"),
    (("sec investigation", "doj investigation", "subpoena", "fraud", "accounting probe"), 4, "regulatory/legal probe"),
    (("restatement", "material weakness", "auditor resign", "accounting issue"), 3, "accounting-control risk"),
    (("share offering", "stock offering", "public offering", "registered direct", "atm offering", "at-the-market offering", "dilution", "selling stockholders"), 3, "dilution/offering risk"),
    (("recall", "fda rejects", "clinical hold", "data breach", "cyberattack"), 3, "event shock risk"),
    (("cuts guidance", "lowers guidance", "misses estimates", "downgrade"), 2, "negative guidance/analyst risk"),
    (("lawsuit", "class action", "settlement"), 2, "litigation risk"),
)

_POSITIVE_RULES: tuple[tuple[tuple[str, ...], int, str], ...] = (
    (("raises guidance", "beats estimates", "record revenue", "profit tops"), 2, "positive earnings/guidance"),
    (("buyback", "repurchase", "dividend increase"), 1, "capital-return news"),
    (("fda approval", "wins approval", "major contract", "partnership", "acquisition"), 1, "positive corporate event"),
    (("upgrade", "price target raised"), 1, "positive analyst action"),
)

_OFFERING_FORMS = {"S-1", "S-3", "424B5", "FWP"}
_EVENT_FORMS = {"8-K", "6-K"}


def classify_research_signals(
    symbol: str,
    news_items: list[AlpacaNewsItem],
    official: OfficialResearch | None,
    now: datetime,
    official_texts: list[str] | None = None,
    block_risk_score: int = 4,
    caution_risk_score: int = 2,
    extra_reason: str = "",
) -> ResearchSymbolSnapshot:
    upper = symbol.upper()
    risk_score = 0
    positive_score = 0
    reasons: list[str] = []
    latest_news = news_items[0] if news_items else None

    for item in news_items[:5]:
        text = f"{item.headline} {item.summary}".lower()
        risk_delta, positive_delta = _score_research_text(text, reasons)
        risk_score += risk_delta
        positive_score += positive_delta

    for text in official_texts or []:
        risk_delta, positive_delta = _score_research_text(text.lower(), reasons)
        risk_score += min(risk_delta, 4)
        positive_score += min(positive_delta, 2)

    latest_filing = official.latest_filings[0] if official and official.latest_filings else None
    if latest_filing:
        age_days = _filing_age_days(latest_filing.filed, now)
        if latest_filing.form in _OFFERING_FORMS and age_days <= 7:
            risk_score += 3
            reasons.append(f"recent SEC offering filing {latest_filing.form}")
        elif latest_filing.form in _EVENT_FORMS and age_days <= 2:
            risk_score += 1
            reasons.append(f"recent SEC event filing {latest_filing.form}")

    if extra_reason:
        reasons.append(extra_reason)

    if official is None and not news_items:
        risk_level = "unavailable"
        allow_entry = True
        if not reasons:
            reasons.append("official/news data unavailable")
    elif risk_score >= block_risk_score:
        risk_level = "blocked"
        allow_entry = False
    elif risk_score >= caution_risk_score:
        risk_level = "caution"
        allow_entry = True
    else:
        risk_level = "ok"
        allow_entry = True

    return ResearchSymbolSnapshot(
        symbol=upper,
        checked_at=now,
        allow_entry=allow_entry,
        risk_level=risk_level,
        risk_score=risk_score,
        positive_score=positive_score,
        reasons=tuple(reasons[:8]),
        news_count=len(news_items),
        filings_count=len(official.latest_filings) if official else 0,
        latest_news_headline=latest_news.headline if latest_news else "",
        latest_news_source=latest_news.source if latest_news else "",
        latest_news_url=latest_news.url if latest_news else "",
        latest_filing_form=latest_filing.form if latest_filing else "",
        latest_filing_date=latest_filing.filed if latest_filing else "",
        latest_filing_url=latest_filing.url if latest_filing else "",
    )


def _is_excluded_news_item(item: AlpacaNewsItem) -> bool:
    text = f"{item.source} {item.url} {item.headline}".lower()
    return any(marker in text for marker in _EXCLUDED_NEWS_MARKERS)


def _filing_age_days(filed: str, now: datetime) -> int:
    try:
        filed_date = datetime.fromisoformat(filed).date()
    except ValueError:
        return 9999
    return max((now.astimezone(KST).date() - filed_date).days, 0)


def _score_research_text(text: str, reasons: list[str]) -> tuple[int, int]:
    risk_score = 0
    positive_score = 0
    for terms, weight, reason in _RISK_RULES:
        if any(term in text for term in terms):
            risk_score += weight
            if reason not in reasons:
                reasons.append(reason)
    for terms, weight, reason in _POSITIVE_RULES:
        if any(term in text for term in terms):
            positive_score += weight
            if reason not in reasons:
                reasons.append(reason)
    return risk_score, positive_score


class YahooRealtimeDataClient:
    def __init__(self) -> None:
        self.client = YahooChartClient(cache_ttl_hours=0)
        self.last_market_times: dict[str, int] = {}

    def latest_quote_ticks(self, symbols: list[str], feed: str = "yahoo") -> list[dict]:
        ticks: list[dict] = []
        for symbol in symbols:
            upper = symbol.upper()
            try:
                snapshot = self.client.intraday_snapshot(upper, interval="1m", range_="1d")
            except DataError:
                continue
            except Exception:
                continue
            market_time = snapshot.get("market_time")
            if isinstance(market_time, int):
                if self.last_market_times.get(upper) == market_time:
                    continue
                self.last_market_times[upper] = market_time
                timestamp = datetime.fromtimestamp(market_time, tz=ZoneInfo("UTC"))
            else:
                timestamp = datetime.now(ZoneInfo("UTC"))
            price = float(snapshot.get("price") or 0.0)
            if price <= 0:
                continue
            ticks.append(
                {
                    "timestamp": timestamp,
                    "symbol": upper,
                    "price": price,
                    "bid": None,
                    "ask": None,
                    "volume": int(snapshot.get("recent_volume") or 0),
                    "source": "yahoo:chart",
                }
            )
        return ticks


class DynamicUniverseScanner:
    def __init__(
        self,
        symbols: list[str],
        config: SurgeScanConfig,
        max_scan_symbols: int = 60,
        provider: str = "yahoo",
        feed: str = "iex",
    ) -> None:
        self.symbols = _dedupe_symbols(symbols)[:max_scan_symbols]
        self.config = config
        self.provider = provider.lower()
        self.feed = feed
        self.yahoo_client = YahooChartClient(cache_ttl_hours=0)
        self.alpaca_client: AlpacaMarketDataClient | None = None
        if self.provider == "alpaca":
            self.alpaca_client = AlpacaMarketDataClient()

    def scan(self) -> list[SurgeCandidate]:
        if self.provider == "alpaca" and self.alpaca_client:
            histories = self.alpaca_client.intraday_bars(self.symbols, feed=self.feed, minutes=120)
            return score_surging_symbols(histories, self.config)

        histories: dict[str, list[IntradayBar]] = {}
        for symbol in self.symbols:
            try:
                bars = self.yahoo_client.intraday_history(symbol, interval="1m", range_="1d")
            except DataError:
                continue
            except Exception:
                continue
            if bars:
                histories[symbol] = bars
        return score_surging_symbols(histories, self.config)


def score_surging_symbols(
    histories: dict[str, list[IntradayBar]],
    config: SurgeScanConfig | None = None,
) -> list[SurgeCandidate]:
    cfg = config or SurgeScanConfig()
    candidates: list[SurgeCandidate] = []
    for symbol, raw_bars in histories.items():
        bars = sorted(raw_bars, key=lambda item: item.timestamp)
        required = cfg.recent_bars + 2
        if len(bars) < required:
            continue
        recent = bars[-cfg.recent_bars :]
        baseline_start = max(0, len(bars) - cfg.recent_bars - cfg.baseline_bars)
        baseline = bars[baseline_start : -cfg.recent_bars]
        if not baseline:
            baseline = bars[: -cfg.recent_bars]
        if not baseline:
            continue

        recent_avg_volume = sum(bar.volume for bar in recent) / len(recent)
        baseline_avg_volume = max(sum(bar.volume for bar in baseline) / len(baseline), 1.0)
        volume_ratio = recent_avg_volume / baseline_avg_volume
        price = float(bars[-1].close)
        recent_dollar_volume = sum(float(bar.close) * float(bar.volume) for bar in recent)
        reference_price = float(bars[-cfg.recent_bars - 1].close)
        short_return = price / reference_price - 1 if reference_price > 0 else 0.0
        first_price = float(bars[0].open or bars[0].close)
        day_return = price / first_price - 1 if first_price > 0 else 0.0

        if price < 5:
            continue
        if volume_ratio < cfg.min_volume_ratio:
            continue
        if recent_dollar_volume < cfg.min_recent_dollar_volume:
            continue
        if short_return < cfg.min_short_return_pct:
            continue

        score = (
            math.log(max(volume_ratio, 1.0))
            + min(recent_dollar_volume / 50_000_000.0, 2.0)
            + max(short_return, 0.0) * 30.0
            + max(day_return, 0.0) * 5.0
        )
        candidates.append(
            SurgeCandidate(
                symbol=symbol.upper(),
                score=score,
                price=price,
                volume_ratio=volume_ratio,
                recent_dollar_volume=recent_dollar_volume,
                short_return=short_return,
                day_return=day_return,
                bars=len(bars),
            )
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[: cfg.max_candidates]


class RealtimePaperLedger:
    def __init__(self) -> None:
        self.open_trades: dict[str, dict] = {}
        self.closed_trades: list[dict] = []

    def record(self, decision: RealtimeDecision) -> dict | None:
        if decision.action == "BUY_PLAN":
            self.open_trades[decision.symbol] = {
                "symbol": decision.symbol,
                "entry_time": decision.timestamp.isoformat(),
                "entry_price": decision.price,
                "qty": decision.qty,
                "entry_notional": decision.qty * decision.price,
                "entry_reason": decision.reason,
                "source": decision.source,
            }
            return None
        if decision.action not in {"SELL_PLAN", "SELL_PARTIAL_PLAN"}:
            return None

        opened = self.open_trades.get(decision.symbol)
        entry_price = float(opened["entry_price"]) if opened else decision.price
        entry_time = str(opened["entry_time"]) if opened else ""
        entry_notional = decision.qty * entry_price
        exit_notional = decision.qty * decision.price
        realized_pnl = exit_notional - entry_notional
        pnl_pct = realized_pnl / entry_notional if entry_notional else 0.0
        trade = {
            "symbol": decision.symbol,
            "entry_time": entry_time,
            "exit_time": decision.timestamp.isoformat(),
            "qty": decision.qty,
            "entry_price": entry_price,
            "exit_price": decision.price,
            "entry_notional": entry_notional,
            "exit_notional": exit_notional,
            "realized_pnl": realized_pnl,
            "pnl_pct": pnl_pct,
            "reason": decision.reason,
            "source": decision.source,
        }
        self.closed_trades.append(trade)
        if opened and decision.action == "SELL_PARTIAL_PLAN":
            opened["qty"] = max(float(opened["qty"]) - decision.qty, 0.0)
        elif opened:
            self.open_trades.pop(decision.symbol, None)
        return trade


def realtime_config_for_mode(risk_mode: str) -> RealtimeConfig:
    normalized = risk_mode.lower()
    if normalized == "aggressive":
        return RealtimeConfig(
            lookback_ticks=5,
            entry_momentum_pct=0.0028,
            exit_momentum_pct=-0.004,
            stop_loss_pct=0.0065,
            take_profit_pct=0.025,
            trailing_stop_pct=0.007,
            max_spread_pct=0.0032,
            max_position_pct=0.40,
            cooldown_ticks=8,
            max_trades_per_day=4,
        )
    if normalized == "balanced":
        return RealtimeConfig(
            lookback_ticks=6,
            entry_momentum_pct=0.0018,
            stop_loss_pct=0.0055,
            take_profit_pct=0.011,
            trailing_stop_pct=0.005,
            max_spread_pct=0.0028,
            max_position_pct=0.40,
            cooldown_ticks=10,
            max_trades_per_day=3,
        )
    if normalized in {"surge", "surge_runner", "runner"}:
        return RealtimeConfig(
            lookback_ticks=6,
            entry_momentum_pct=0.0034,
            exit_momentum_pct=-0.006,
            stop_loss_pct=0.008,
            take_profit_pct=0.070,
            trailing_stop_pct=0.012,
            max_spread_pct=0.0038,
            max_position_pct=0.40,
            cooldown_ticks=8,
            max_trades_per_day=4,
            partial_take_profit_pct=0.015,
            partial_take_profit_fraction=0.40,
            breakeven_after_pct=0.020,
            breakeven_offset_pct=0.001,
            execution_slippage_bps=3.0,
        )
    if normalized in {"hybrid", "hybrid_runner", "guarded_runner"}:
        return RealtimeConfig(
            lookback_ticks=6,
            entry_momentum_pct=0.0015,
            exit_momentum_pct=-0.003,
            stop_loss_pct=0.005,
            take_profit_pct=0.012,
            trailing_stop_pct=0.005,
            max_spread_pct=0.0025,
            max_position_pct=0.40,
            cooldown_ticks=10,
            max_trades_per_day=3,
            partial_take_profit_pct=0.006,
            partial_take_profit_fraction=0.35,
            breakeven_after_pct=0.008,
            breakeven_offset_pct=0.0005,
            execution_slippage_bps=2.5,
            min_tick_volume=1,
            max_tick_age_seconds=75,
        )
    if normalized in {"profit", "profit_runner", "profit_vwap_runner"}:
        return RealtimeConfig(
            lookback_ticks=6,
            entry_momentum_pct=0.0015,
            exit_momentum_pct=-0.003,
            stop_loss_pct=0.005,
            take_profit_pct=0.010,
            trailing_stop_pct=0.005,
            max_spread_pct=0.0025,
            max_position_pct=0.40,
            cooldown_ticks=10,
            max_trades_per_day=3,
            execution_slippage_bps=2.0,
            min_tick_volume=1,
            max_tick_age_seconds=75,
        )
    return RealtimeConfig(
        lookback_ticks=6,
        entry_momentum_pct=0.0015,
        exit_momentum_pct=-0.003,
        stop_loss_pct=0.005,
        take_profit_pct=0.010,
        trailing_stop_pct=0.005,
        max_spread_pct=0.0025,
        max_position_pct=0.40,
        cooldown_ticks=10,
        max_trades_per_day=3,
        execution_slippage_bps=2.0,
        min_tick_volume=1,
        max_tick_age_seconds=75,
    )


def run_realtime_paper_session(config: RealtimeSessionConfig) -> Path:
    until = config.until or next_kst_9am()
    session_dir = _new_session_dir(config.output_root)
    paths = _session_paths(session_dir)
    base_model_config = realtime_config_for_mode(config.risk_mode)
    model = RealtimeReactiveModel(cash=config.cash, config=base_model_config)
    controller = AdaptiveRiskController(base_model_config)
    ledger = RealtimePaperLedger()
    latest_prices: dict[str, float] = {}
    base_symbols = _dedupe_symbols(config.symbols)
    active_symbols = list(base_symbols)
    latest_surge_candidates: list[SurgeCandidate] = []
    last_scan_at: datetime | None = None
    scan_count = 0
    scanner = _build_dynamic_scanner(config, base_symbols)
    market_filter = MarketFilter(
        config.market_symbols,
        lookback_ticks=config.market_lookback_ticks,
        min_momentum_pct=config.min_market_momentum_pct,
    ) if config.market_filter else None
    market_snapshot = MarketSnapshot(
        True,
        "market filter enabled; waiting for US regular market" if config.market_filter else "market filter disabled",
        {},
    )
    research_monitor = ResearchRiskMonitor(
        provider=config.provider,
        feed=config.feed,
        news_lookback_minutes=config.research_news_lookback_minutes,
        block_risk_score=config.research_block_risk_score,
        caution_risk_score=config.research_caution_risk_score,
        sec_cache_ttl_seconds=config.sec_cache_ttl_seconds,
    ) if config.research_filter else None
    research_snapshot = ResearchSnapshot(
        config.research_filter,
        datetime.now(KST),
        "research filter enabled; waiting for US regular market" if config.research_filter else "research filter disabled",
        {},
    )
    last_research_at: datetime | None = None
    started_at = datetime.now(KST)
    status = "running"
    last_error = ""
    total_ticks = 0
    total_decisions = 0
    client: AlpacaMarketDataClient | YahooRealtimeDataClient | None = None
    provider = config.provider.lower()
    if provider not in {"alpaca", "yahoo"}:
        raise ValueError("provider must be alpaca or yahoo")
    market_hours_calendar = AlpacaMarketHoursCalendar(provider)
    market_hours_snapshot = MarketHoursSnapshot(
        False,
        "market calendar enabled; waiting for first check",
        source="alpaca_calendar" if provider == "alpaca" else "fallback_regular_hours",
    )

    _init_session_files(paths)
    _write_json(
        paths["meta"],
        {
            "started_at": started_at.isoformat(),
            "until": until.isoformat(),
            "symbols": base_symbols,
            "feed": config.feed,
            "poll_seconds": config.poll_seconds,
            "risk_mode": config.risk_mode,
            "starting_cash": config.cash,
            "provider": provider,
            "dynamic_universe": config.dynamic_universe,
            "scan_symbols": scanner.symbols if scanner else [],
            "dynamic_max_symbols": config.dynamic_max_symbols,
            "scan_interval_seconds": config.scan_interval_seconds,
            "market_filter": config.market_filter,
            "market_symbols": list(config.market_symbols),
            "market_lookback_ticks": config.market_lookback_ticks,
            "min_market_momentum_pct": config.min_market_momentum_pct,
            "market_hours_source": market_hours_snapshot.source,
            "research_filter": config.research_filter,
            "research_interval_seconds": config.research_interval_seconds,
            "research_max_symbols": config.research_max_symbols,
            "research_news_lookback_minutes": config.research_news_lookback_minutes,
            "research_block_risk_score": config.research_block_risk_score,
            "research_caution_risk_score": config.research_caution_risk_score,
            "paper_mode": "local_simulation_no_real_orders",
        },
    )

    try:
        while datetime.now(KST) < until:
            now = datetime.now(KST)
            if client is None:
                if provider == "yahoo":
                    client = YahooRealtimeDataClient()
                    status = "running_yahoo_fallback"
                    last_error = "Alpaca API 키가 없어 Yahoo Finance 보조 피드로 기록합니다. 실제 주문 판단용 공식 브로커 피드가 아닙니다."
                    _append_event(paths["events"], now, "YAHOO_FALLBACK", last_error)
                elif not alpaca_credentials_available():
                    status = "waiting_for_credentials"
                    last_error = "ALPACA_KEY_ID/ALPACA_SECRET_KEY가 없어 데이터 수집 대기 중입니다."
                    _write_summary(
                        paths,
                        session_dir,
                        model,
                        ledger,
                        latest_prices,
                        config,
                        base_model_config,
                        started_at,
                        now,
                        until,
                        status,
                        last_error,
                        total_ticks,
                        total_decisions,
                        active_symbols,
                        latest_surge_candidates,
                        scan_count,
                        market_snapshot,
                        research_snapshot,
                        market_hours_snapshot,
                    )
                    if not config.wait_for_credentials:
                        raise AlpacaError(last_error)
                    sleep(max(config.credential_check_seconds, 5))
                    continue
                elif provider == "alpaca":
                    client = AlpacaMarketDataClient()
                    status = "running"
                    last_error = ""

            market_hours_snapshot = market_hours_calendar.snapshot(now)
            if not market_hours_snapshot.is_open:
                equity, market_value = model.mark_to_market(latest_prices)
                _append_equity(paths["equity"], now, model, equity, market_value, config.cash, latest_prices)
                _write_summary(
                    paths,
                    session_dir,
                    model,
                    ledger,
                    latest_prices,
                    config,
                    base_model_config,
                    started_at,
                    now,
                    until,
                    f"{status}:waiting_for_us_regular_market",
                    last_error,
                    total_ticks,
                    total_decisions,
                    active_symbols,
                    latest_surge_candidates,
                    scan_count,
                    market_snapshot,
                    research_snapshot,
                    market_hours_snapshot,
                )
                sleep(min(max(config.poll_seconds, 1) * 6, 60))
                continue

            if scanner and _scan_due(now, last_scan_at, config.scan_interval_seconds):
                latest_surge_candidates = scanner.scan()
                last_scan_at = now
                scan_count += 1
                active_symbols = _select_active_symbols(
                    base_symbols,
                    latest_surge_candidates,
                    list(model.positions.keys()),
                    config.dynamic_max_symbols,
                )
                _append_universe_scan(paths["universe"], now, latest_surge_candidates, active_symbols)
                if latest_surge_candidates:
                    _append_event(
                        paths["events"],
                        now,
                        "DYNAMIC_UNIVERSE_SCAN",
                        f"active={','.join(active_symbols)} candidates={','.join(item.symbol for item in latest_surge_candidates)}",
                    )

            if research_monitor and _scan_due(now, last_research_at, config.research_interval_seconds):
                research_symbols = _select_research_symbols(
                    active_symbols,
                    latest_surge_candidates,
                    list(model.positions.keys()),
                    config.research_max_symbols,
                )
                try:
                    research_snapshot = research_monitor.refresh(research_symbols, now)
                except Exception as exc:
                    research_snapshot = ResearchSnapshot(True, now, f"research refresh failed: {exc}", {})
                    _append_event(paths["events"], now, "RESEARCH_ERROR", str(exc))
                last_research_at = now
                _append_research_snapshot(paths["research"], research_snapshot)
                if research_snapshot.reason != "official/news filter clear":
                    _append_event(paths["events"], now, "RESEARCH_FILTER", research_snapshot.reason)

            try:
                quote_symbols = _dedupe_symbols([*active_symbols, *(list(config.market_symbols) if config.market_filter else [])])
                raw_ticks = client.latest_quote_ticks(quote_symbols, feed=config.feed)
            except AlpacaError as exc:
                last_error = str(exc)
                _append_event(paths["events"], now, "DATA_ERROR", last_error)
                sleep(max(config.poll_seconds, 1))
                continue

            ticks = [
                PriceTick(
                    timestamp=item["timestamp"],
                    symbol=item["symbol"],
                    price=item["price"],
                    bid=item.get("bid"),
                    ask=item.get("ask"),
                    volume=item.get("volume"),
                    source=item["source"],
                )
                for item in raw_ticks
            ]
            ticks.sort(key=lambda item: (item.timestamp, item.symbol))
            total_ticks += len(ticks)

            for tick in ticks:
                latest_prices[tick.symbol] = tick.price
                _append_tick(paths["ticks"], tick)
                if market_filter and tick.symbol in market_filter.symbols:
                    market_filter.update(tick)

            market_snapshot = market_filter.snapshot() if market_filter else MarketSnapshot(True, "market filter disabled", {})

            for tick in ticks:
                if tick.symbol not in active_symbols:
                    continue
                _append_symbol_snapshot(paths["symbols"], now, tick, model, market_snapshot, research_snapshot)
                allow_entry = market_snapshot.risk_on and research_snapshot.allows_entry(tick.symbol)
                decisions = model.on_tick(tick, allow_entry=allow_entry)
                for decision in decisions:
                    total_decisions += 1
                    _append_decision(paths["decisions"], decision)
                    closed_trade = ledger.record(decision)
                    if closed_trade:
                        _append_row(paths["trades"], _trade_fields(), _format_trade_row(closed_trade))
                        event = controller.observe_closed_trade(model, decision.timestamp, float(closed_trade["pnl_pct"]))
                        if event:
                            _append_adaptation(paths["adaptations"], event)

            equity, market_value = model.mark_to_market(latest_prices)
            event = controller.observe_equity(model, now, config.cash, equity)
            if event:
                _append_adaptation(paths["adaptations"], event)
            _append_equity(paths["equity"], now, model, equity, market_value, config.cash, latest_prices)
            _write_summary(
                paths,
                session_dir,
                model,
                ledger,
                latest_prices,
                config,
                base_model_config,
                started_at,
                now,
                until,
                status,
                last_error,
                total_ticks,
                total_decisions,
                active_symbols,
                latest_surge_candidates,
                scan_count,
                market_snapshot,
                research_snapshot,
                market_hours_snapshot,
            )
            sleep(max(config.poll_seconds, 1))
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        ended_at = datetime.now(KST)
        if status in {"running", "running_yahoo_fallback"}:
            status = "completed" if ended_at >= until else "stopped"
        _write_summary(
            paths,
            session_dir,
            model,
            ledger,
            latest_prices,
            config,
            base_model_config,
            started_at,
            ended_at,
            until,
            status,
            last_error,
            total_ticks,
            total_decisions,
            active_symbols,
            latest_surge_candidates,
            scan_count,
            market_snapshot,
            research_snapshot,
            market_hours_snapshot,
        )
    return session_dir


def summarize_realtime_session(session_dir: Path | None = None, output_root: Path = Path("reports")) -> str:
    resolved = session_dir or latest_realtime_session_dir(output_root)
    if resolved is None:
        return "요약할 realtime paper trading 세션이 없습니다."
    paths = _session_paths(resolved)
    if not paths["summary"].exists():
        return f"요약 파일이 없습니다: {resolved}"
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    _write_report(paths, summary)

    lines = [
        "## 실시간 Paper Trading 요약",
        "",
        "| 항목 | 값 |",
        "| --- | ---: |",
        f"| 상태 | {summary.get('status', '-')} |",
        f"| 시작 시각 | {summary.get('started_at', '-')} |",
        f"| 종료/요약 시각 | {summary.get('ended_at', '-')} |",
        f"| 목표 종료 시각 | {summary.get('until', '-')} |",
        f"| 대상 종목 | {', '.join(summary.get('symbols', []))} |",
        f"| 시작 총자산 | {_money(float(summary.get('start_equity', 0.0)))} |",
        f"| 현재/최종 총자산 | {_money(float(summary.get('end_equity', 0.0)))} |",
        f"| 누적 수익률 | {float(summary.get('total_return', 0.0)):.2%} |",
        f"| 최대 낙폭 | {float(summary.get('max_drawdown', 0.0)):.2%} |",
        f"| 수집 tick | {int(summary.get('ticks', 0)):,}개 |",
        f"| 매수/매도 계획 | {int(summary.get('decisions', 0)):,}건 |",
        f"| 청산 거래 | {int(summary.get('closed_trades', 0)):,}건 |",
        f"| 보유 포지션 | {int(summary.get('open_positions', 0)):,}개 |",
    ]
    lines.insert(10, f"| 현재 감시 종목 | {', '.join(summary.get('active_symbols', summary.get('symbols', [])))} |")
    lines.insert(11, f"| 동적 스캔 횟수 | {int(summary.get('scan_count', 0)):,}회 |")
    lines.insert(12, f"| 시장 필터 | {'통과' if summary.get('market_risk_on', True) else '차단'} |")
    lines.insert(13, f"| 시장 필터 사유 | {summary.get('market_reason', '-')} |")
    lines.insert(14, f"| 장시간 캘린더 | {summary.get('market_hours_reason', '-')} |")
    lines.insert(15, f"| 데이터 피드 범위 | {summary.get('data_feed_scope', '-')} |")
    lines.insert(16, f"| 뉴스/공시 보조지표 | {'사용' if summary.get('research_filter') else '미사용'} |")
    lines.insert(17, f"| 뉴스/공시 상태 | {summary.get('research_reason', '-')} |")
    if summary.get("last_error"):
        lines.append(f"| 마지막 경고 | {summary['last_error']} |")

    trades = _read_csv_dicts(paths["trades"])
    lines.extend(["", _format_closed_trades_table(trades)])
    positions = summary.get("positions", [])
    lines.extend(["", _format_positions_table(positions)])
    adaptations = _read_csv_dicts(paths["adaptations"])
    lines.extend(["", _format_adaptations_table(adaptations)])
    lines.extend(["", _format_surge_candidates_table(summary.get("surge_candidates", []))])
    lines.extend(
        [
            "",
            f"- 세션 폴더: {resolved}",
            f"- 총자산 그래프: {summary.get('equity_chart') or '-'}",
            f"- 거래 CSV: {paths['trades']}",
            f"- 리포트: {paths['report']}",
        ]
    )
    lines.append(f"- 동적 종목 스캔 CSV: {paths['universe']}")
    lines.append(f"- 종목별 실시간 데이터 CSV: {paths['symbols']}")
    if "research" in paths:
        lines.append(f"- 뉴스/공시 보조지표 CSV: {paths['research']}")
    return "\n".join(lines)


def latest_realtime_session_dir(output_root: Path = Path("reports")) -> Path | None:
    if not output_root.exists():
        return None
    sessions = [path for path in output_root.glob("realtime_paper_*") if path.is_dir()]
    if not sessions:
        return None
    return max(sessions, key=lambda item: item.stat().st_mtime)


def parse_session_until(value: str | None) -> datetime:
    if not value:
        return next_kst_9am()
    stripped = value.strip()
    if len(stripped) <= 5 and ":" in stripped:
        hour, minute = [int(part) for part in stripped.split(":", 1)]
        return _next_kst_time(time(hour=hour, minute=minute))
    parsed = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=KST)
    return parsed.astimezone(KST)


def next_kst_9am(now: datetime | None = None) -> datetime:
    return _next_kst_time(time(hour=9, minute=0), now=now)


def _is_us_regular_market(timestamp: datetime) -> bool:
    ny_time = timestamp.astimezone(NEW_YORK)
    if ny_time.weekday() >= 5:
        return False
    regular_open = ny_time.replace(hour=9, minute=30, second=0, microsecond=0)
    regular_close = ny_time.replace(hour=16, minute=0, second=0, microsecond=0)
    return regular_open <= ny_time < regular_close


def _parse_alpaca_calendar_datetime(item: dict, fallback_date: str, key: str) -> datetime | None:
    value = item.get(key) or item.get(f"session_{key}") or item.get(f"{key}_time")
    if not value:
        return None
    text = str(value)
    if "T" in text:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=NEW_YORK)
        return parsed.astimezone(NEW_YORK)
    date_text = str(item.get("date") or fallback_date)[:10]
    parsed_date = date.fromisoformat(date_text)
    parsed_time = time.fromisoformat(text)
    return datetime.combine(parsed_date, parsed_time, tzinfo=NEW_YORK)


def _build_dynamic_scanner(config: RealtimeSessionConfig, base_symbols: list[str]) -> DynamicUniverseScanner | None:
    if not config.dynamic_universe:
        return None
    scan_symbols = config.scan_symbols or base_symbols
    scan_config = SurgeScanConfig(
        min_volume_ratio=config.min_volume_ratio,
        min_recent_dollar_volume=config.min_recent_dollar_volume,
        min_short_return_pct=config.min_short_return_pct,
        max_candidates=config.top_surging_symbols,
    )
    return DynamicUniverseScanner(
        scan_symbols,
        config=scan_config,
        max_scan_symbols=config.scan_max_symbols,
        provider=config.provider,
        feed=config.feed,
    )


def _scan_due(now: datetime, last_scan_at: datetime | None, scan_interval_seconds: int) -> bool:
    if last_scan_at is None:
        return True
    return (now - last_scan_at).total_seconds() >= max(scan_interval_seconds, 30)


def _select_active_symbols(
    base_symbols: list[str],
    candidates: list[SurgeCandidate],
    position_symbols: list[str],
    max_symbols: int,
) -> list[str]:
    selected: list[str] = []
    limit = max(max_symbols, len(base_symbols), 1)
    for symbol in [*base_symbols, *position_symbols, *(item.symbol for item in candidates)]:
        upper = symbol.upper()
        if upper not in selected:
            selected.append(upper)
        if len(selected) >= limit:
            break
    return selected


def _select_research_symbols(
    active_symbols: list[str],
    candidates: list[SurgeCandidate],
    position_symbols: list[str],
    max_symbols: int,
) -> list[str]:
    selected: list[str] = []
    for symbol in [*position_symbols, *(item.symbol for item in candidates), *active_symbols]:
        upper = symbol.upper()
        if upper and upper not in selected:
            selected.append(upper)
        if len(selected) >= max(max_symbols, 1):
            break
    return selected


def _dedupe_symbols(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for symbol in symbols:
        upper = symbol.upper()
        if upper and upper not in seen and not upper.startswith("^"):
            seen.add(upper)
            result.append(upper)
    return result


def _next_kst_time(target: time, now: datetime | None = None) -> datetime:
    current = (now or datetime.now(KST)).astimezone(KST)
    candidate = current.replace(hour=target.hour, minute=target.minute, second=0, microsecond=0)
    if candidate <= current:
        candidate += timedelta(days=1)
    return candidate


def _new_session_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stem = f"realtime_paper_{datetime.now(KST).strftime('%Y%m%d_%H%M%S')}"
    for suffix in ["", *[f"_{idx}" for idx in range(1, 100)]]:
        path = output_root / f"{stem}{suffix}"
        try:
            path.mkdir(parents=True, exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique realtime session directory for {stem}")


def _session_paths(session_dir: Path) -> dict[str, Path]:
    return {
        "meta": session_dir / "session_meta.json",
        "ticks": session_dir / "ticks.jsonl",
        "decisions": session_dir / "decisions.csv",
        "equity": session_dir / "equity.csv",
        "trades": session_dir / "closed_trades.csv",
        "adaptations": session_dir / "adaptations.csv",
        "events": session_dir / "events.csv",
        "universe": session_dir / "dynamic_universe.csv",
        "symbols": session_dir / "symbol_snapshots.csv",
        "research": session_dir / "research_snapshots.csv",
        "summary": session_dir / "summary.json",
        "chart": session_dir / "equity.png",
        "report": session_dir / "report.md",
    }


def _init_session_files(paths: dict[str, Path]) -> None:
    _write_header(paths["decisions"], _decision_fields())
    _write_header(paths["equity"], _equity_fields())
    _write_header(paths["trades"], _trade_fields())
    _write_header(paths["adaptations"], _adaptation_fields())
    _write_header(paths["events"], ["timestamp", "event", "detail"])
    _write_header(paths["universe"], _universe_fields())
    _write_header(paths["symbols"], _symbol_snapshot_fields())
    _write_header(paths["research"], _research_fields())


def _write_summary(
    paths: dict[str, Path],
    session_dir: Path,
    model: RealtimeReactiveModel,
    ledger: RealtimePaperLedger,
    latest_prices: dict[str, float],
    session_config: RealtimeSessionConfig,
    base_model_config: RealtimeConfig,
    started_at: datetime,
    ended_at: datetime,
    until: datetime,
    status: str,
    last_error: str,
    total_ticks: int,
    total_decisions: int,
    active_symbols: list[str] | None = None,
    surge_candidates: list[SurgeCandidate] | None = None,
    scan_count: int = 0,
    market_snapshot: MarketSnapshot | None = None,
    research_snapshot: ResearchSnapshot | None = None,
    market_hours_snapshot: MarketHoursSnapshot | None = None,
) -> None:
    equity, market_value = model.mark_to_market(latest_prices)
    equity_rows = _read_csv_dicts(paths["equity"])
    max_drawdown = _max_drawdown([float(row["equity"]) for row in equity_rows] + [equity])
    realized_pnl = sum(float(trade["realized_pnl"]) for trade in ledger.closed_trades)
    positions = model.position_rows(latest_prices)
    unrealized_pnl = sum(float(row["unrealized_pnl"]) for row in positions)
    chart = _write_realtime_equity_chart(paths["chart"], equity_rows)
    market = market_snapshot or MarketSnapshot(True, "market filter not reported", {})
    research = research_snapshot or ResearchSnapshot(False, ended_at, "research filter not reported", {})
    market_hours = market_hours_snapshot or MarketHoursSnapshot(False, "market hours not reported")
    blocked_research_symbols = [
        symbol for symbol, item in research.by_symbol.items() if item.risk_level == "blocked"
    ]
    caution_research_symbols = [
        symbol for symbol, item in research.by_symbol.items() if item.risk_level == "caution"
    ]
    summary = {
        "session_dir": str(session_dir),
        "status": status,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "until": until.isoformat(),
        "symbols": session_config.symbols,
        "active_symbols": active_symbols or session_config.symbols,
        "provider": session_config.provider,
        "feed": session_config.feed,
        "risk_mode": session_config.risk_mode,
        "dynamic_universe": session_config.dynamic_universe,
        "scan_count": scan_count,
        "surge_candidates": [asdict(item) for item in (surge_candidates or [])],
        "market_risk_on": market.risk_on,
        "market_reason": market.reason,
        "market_momentums": market.momentums,
        "market_hours_open": market_hours.is_open,
        "market_hours_reason": market_hours.reason,
        "market_open_at": market_hours.open_at.isoformat() if market_hours.open_at else "",
        "market_close_at": market_hours.close_at.isoformat() if market_hours.close_at else "",
        "market_hours_source": market_hours.source,
        "data_feed_scope": _feed_scope(session_config.feed),
        "data_feed_limitation": _feed_limitation(session_config.feed),
        "research_filter": session_config.research_filter,
        "research_reason": research.reason,
        "research_checked_at": research.checked_at.isoformat(),
        "research_blocked_symbols": blocked_research_symbols,
        "research_caution_symbols": caution_research_symbols,
        "research_by_symbol": [asdict(item) for item in research.by_symbol.values()],
        "start_equity": session_config.cash,
        "end_equity": equity,
        "cash": model.cash,
        "market_value": market_value,
        "total_return": equity / session_config.cash - 1 if session_config.cash else 0.0,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "max_drawdown": max_drawdown,
        "ticks": total_ticks,
        "decisions": total_decisions,
        "closed_trades": len(ledger.closed_trades),
        "open_positions": len(positions),
        "positions": positions,
        "model_config": asdict(model.config),
        "base_model_config": asdict(base_model_config),
        "last_error": last_error,
        "equity_chart": str(chart) if chart else "",
    }
    _write_json(paths["summary"], summary)
    _write_report(paths, summary)


def _write_report(paths: dict[str, Path], summary: dict) -> None:
    trades = _read_csv_dicts(paths["trades"])
    adaptations = _read_csv_dicts(paths["adaptations"])
    content = "\n".join(
        [
            "# 실시간 Paper Trading 리포트",
            "",
            f"- 상태: {summary.get('status', '-')}",
            f"- 시작: {summary.get('started_at', '-')}",
            f"- 종료/요약: {summary.get('ended_at', '-')}",
            f"- 대상 종목: {', '.join(summary.get('symbols', []))}",
            f"- 총자산: {_money(float(summary.get('end_equity', 0.0)))}",
            f"- 누적 수익률: {float(summary.get('total_return', 0.0)):.2%}",
            f"- 최대 낙폭: {float(summary.get('max_drawdown', 0.0)):.2%}",
            f"- 뉴스/공시 보조지표: {summary.get('research_reason', '-')}",
            f"- 그래프: {summary.get('equity_chart') or '-'}",
            "",
            _format_closed_trades_table(trades),
            "",
            _format_positions_table(summary.get("positions", [])),
            "",
            _format_adaptations_table(adaptations),
            "",
            "실제 주문은 전송하지 않은 로컬 paper trading 기록입니다.",
        ]
    )
    paths["report"].write_text(content, encoding="utf-8")


def _append_tick(path: Path, tick: PriceTick) -> None:
    item = {
        "timestamp": tick.timestamp.isoformat(),
        "symbol": tick.symbol,
        "price": tick.price,
        "bid": tick.bid,
        "ask": tick.ask,
        "volume": tick.volume,
        "source": tick.source,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _append_decision(path: Path, decision: RealtimeDecision) -> None:
    _append_row(
        path,
        _decision_fields(),
        {
            "timestamp": decision.timestamp.isoformat(),
            "symbol": decision.symbol,
            "action": decision.action,
            "price": f"{decision.price:.6f}",
            "qty": f"{decision.qty:.8f}",
            "notional": f"{decision.notional:.2f}",
            "reason": decision.reason,
            "source": decision.source,
        },
    )


def _append_equity(
    path: Path,
    timestamp: datetime,
    model: RealtimeReactiveModel,
    equity: float,
    market_value: float,
    start_equity: float,
    latest_prices: dict[str, float],
) -> None:
    _append_row(
        path,
        _equity_fields(),
        {
            "timestamp": timestamp.isoformat(),
            "cash": f"{model.cash:.2f}",
            "market_value": f"{market_value:.2f}",
            "equity": f"{equity:.2f}",
            "total_return": f"{(equity / start_equity - 1 if start_equity else 0.0):.8f}",
            "open_positions": str(len(model.positions)),
            "latest_prices": json.dumps(latest_prices, ensure_ascii=False, sort_keys=True),
        },
    )


def _append_adaptation(path: Path, event: AdaptiveEvent) -> None:
    _append_row(
        path,
        _adaptation_fields(),
        {
            "timestamp": event.timestamp.isoformat(),
            "event": event.event,
            "detail": event.detail,
            "entry_momentum_pct": f"{event.entry_momentum_pct:.8f}",
            "max_position_pct": f"{event.max_position_pct:.8f}",
            "max_spread_pct": f"{event.max_spread_pct:.8f}",
            "cooldown_ticks": str(event.cooldown_ticks),
        },
    )


def _append_event(path: Path, timestamp: datetime, event: str, detail: str) -> None:
    _append_row(path, ["timestamp", "event", "detail"], {"timestamp": timestamp.isoformat(), "event": event, "detail": detail})


def _append_universe_scan(
    path: Path,
    timestamp: datetime,
    candidates: list[SurgeCandidate],
    active_symbols: list[str],
) -> None:
    active = set(active_symbols)
    for item in candidates:
        _append_row(
            path,
            _universe_fields(),
            {
                "timestamp": timestamp.isoformat(),
                "symbol": item.symbol,
                "selected": str(item.symbol in active),
                "score": f"{item.score:.8f}",
                "price": f"{item.price:.6f}",
                "volume_ratio": f"{item.volume_ratio:.4f}",
                "recent_dollar_volume": f"{item.recent_dollar_volume:.2f}",
                "short_return": f"{item.short_return:.8f}",
                "day_return": f"{item.day_return:.8f}",
                "bars": str(item.bars),
            },
        )


def _append_research_snapshot(path: Path, snapshot: ResearchSnapshot) -> None:
    for item in snapshot.by_symbol.values():
        _append_row(
            path,
            _research_fields(),
            {
                "timestamp": snapshot.checked_at.isoformat(),
                "symbol": item.symbol,
                "allow_entry": str(item.allow_entry),
                "risk_level": item.risk_level,
                "risk_score": str(item.risk_score),
                "positive_score": str(item.positive_score),
                "reasons": "; ".join(item.reasons),
                "news_count": str(item.news_count),
                "filings_count": str(item.filings_count),
                "latest_news_headline": item.latest_news_headline,
                "latest_news_source": item.latest_news_source,
                "latest_news_url": item.latest_news_url,
                "latest_filing_form": item.latest_filing_form,
                "latest_filing_date": item.latest_filing_date,
                "latest_filing_url": item.latest_filing_url,
            },
        )


def _append_symbol_snapshot(
    path: Path,
    timestamp: datetime,
    tick: PriceTick,
    model: RealtimeReactiveModel,
    market_snapshot: MarketSnapshot,
    research_snapshot: ResearchSnapshot,
) -> None:
    spread_pct = _tick_spread_pct(tick)
    momentum = model._short_momentum(tick.symbol)
    position = model.positions.get(tick.symbol)
    research = research_snapshot.symbol_snapshot(tick.symbol)
    _append_row(
        path,
        _symbol_snapshot_fields(),
        {
            "timestamp": timestamp.isoformat(),
            "tick_timestamp": tick.timestamp.isoformat(),
            "symbol": tick.symbol,
            "price": f"{tick.price:.6f}",
            "bid": "" if tick.bid is None else f"{tick.bid:.6f}",
            "ask": "" if tick.ask is None else f"{tick.ask:.6f}",
            "spread_pct": f"{spread_pct:.8f}",
            "short_momentum": f"{momentum:.8f}",
            "volume": "" if tick.volume is None else str(tick.volume),
            "source": tick.source,
            "has_position": str(position is not None),
            "position_qty": "" if position is None else f"{position.qty:.8f}",
            "entry_price": "" if position is None else f"{position.entry_price:.6f}",
            "market_risk_on": str(market_snapshot.risk_on),
            "market_reason": market_snapshot.reason,
            "research_allow_entry": "" if research is None else str(research.allow_entry),
            "research_risk_level": "" if research is None else research.risk_level,
            "research_risk_score": "" if research is None else str(research.risk_score),
            "research_positive_score": "" if research is None else str(research.positive_score),
            "research_reason": "" if research is None else "; ".join(research.reasons),
        },
    )


def _write_header(path: Path, fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()


def _append_row(path: Path, fields: list[str], row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writerow({field: row.get(field, "") for field in fields})


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _decision_fields() -> list[str]:
    return ["timestamp", "symbol", "action", "price", "qty", "notional", "reason", "source"]


def _equity_fields() -> list[str]:
    return ["timestamp", "cash", "market_value", "equity", "total_return", "open_positions", "latest_prices"]


def _trade_fields() -> list[str]:
    return [
        "symbol",
        "entry_time",
        "exit_time",
        "qty",
        "entry_price",
        "exit_price",
        "entry_notional",
        "exit_notional",
        "realized_pnl",
        "pnl_pct",
        "reason",
        "source",
    ]


def _adaptation_fields() -> list[str]:
    return [
        "timestamp",
        "event",
        "detail",
        "entry_momentum_pct",
        "max_position_pct",
        "max_spread_pct",
        "cooldown_ticks",
    ]


def _universe_fields() -> list[str]:
    return [
        "timestamp",
        "symbol",
        "selected",
        "score",
        "price",
        "volume_ratio",
        "recent_dollar_volume",
        "short_return",
        "day_return",
        "bars",
    ]


def _symbol_snapshot_fields() -> list[str]:
    return [
        "timestamp",
        "tick_timestamp",
        "symbol",
        "price",
        "bid",
        "ask",
        "spread_pct",
        "short_momentum",
        "volume",
        "source",
        "has_position",
        "position_qty",
        "entry_price",
        "market_risk_on",
        "market_reason",
        "research_allow_entry",
        "research_risk_level",
        "research_risk_score",
        "research_positive_score",
        "research_reason",
    ]


def _research_fields() -> list[str]:
    return [
        "timestamp",
        "symbol",
        "allow_entry",
        "risk_level",
        "risk_score",
        "positive_score",
        "reasons",
        "news_count",
        "filings_count",
        "latest_news_headline",
        "latest_news_source",
        "latest_news_url",
        "latest_filing_form",
        "latest_filing_date",
        "latest_filing_url",
    ]


def _format_trade_row(trade: dict) -> dict:
    return {
        "symbol": trade["symbol"],
        "entry_time": trade["entry_time"],
        "exit_time": trade["exit_time"],
        "qty": f"{float(trade['qty']):.8f}",
        "entry_price": f"{float(trade['entry_price']):.6f}",
        "exit_price": f"{float(trade['exit_price']):.6f}",
        "entry_notional": f"{float(trade['entry_notional']):.2f}",
        "exit_notional": f"{float(trade['exit_notional']):.2f}",
        "realized_pnl": f"{float(trade['realized_pnl']):.2f}",
        "pnl_pct": f"{float(trade['pnl_pct']):.8f}",
        "reason": trade["reason"],
        "source": trade["source"],
    }


def _format_closed_trades_table(trades: list[dict], limit: int = 30) -> str:
    if not trades:
        return "## 청산 거래 내역\n\n청산된 거래가 아직 없습니다."
    lines = [
        "## 청산 거래 내역",
        "",
        "| 종목 | 진입 | 청산 | 수량 | 진입가 | 청산가 | 실현손익 | 수익률 | 사유 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for trade in trades[-limit:]:
        lines.append(
            f"| {trade['symbol']} | {_short_time(trade['entry_time'])} | {_short_time(trade['exit_time'])} | "
            f"{float(trade['qty']):,.4f} | {_money(float(trade['entry_price']))} | "
            f"{_money(float(trade['exit_price']))} | {_money(float(trade['realized_pnl']))} | "
            f"{float(trade['pnl_pct']):.2%} | {trade['reason']} |"
        )
    return "\n".join(lines)


def _format_positions_table(positions: list[dict], limit: int = 30) -> str:
    if not positions:
        return "## 현재 보유 포지션\n\n보유 중인 포지션이 없습니다."
    lines = [
        "## 현재 보유 포지션",
        "",
        "| 종목 | 수량 | 진입가 | 현재가 | 평가금액 | 미실현손익 | 수익률 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in positions[:limit]:
        lines.append(
            f"| {item['symbol']} | {float(item['qty']):,.4f} | {_money(float(item['entry_price']))} | "
            f"{_money(float(item['latest_price']))} | {_money(float(item['market_value']))} | "
            f"{_money(float(item['unrealized_pnl']))} | {float(item['unrealized_return']):.2%} |"
        )
    return "\n".join(lines)


def _format_adaptations_table(adaptations: list[dict], limit: int = 20) -> str:
    if not adaptations:
        return "## 모델 자동 조정 내역\n\n자동 조정은 아직 발생하지 않았습니다."
    lines = [
        "## 모델 자동 조정 내역",
        "",
        "| 시각 | 이벤트 | 내용 | 진입 모멘텀 | 종목 비중 | 스프레드 한도 | 쿨다운 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in adaptations[-limit:]:
        lines.append(
            f"| {_short_time(item['timestamp'])} | {item['event']} | {item['detail']} | "
            f"{float(item['entry_momentum_pct']):.3%} | {float(item['max_position_pct']):.2%} | "
            f"{float(item['max_spread_pct']):.3%} | {int(item['cooldown_ticks'])} |"
        )
    return "\n".join(lines)


def _format_surge_candidates_table(candidates: list[dict], limit: int = 20) -> str:
    if not candidates:
        return "## 거래량/거래대금 급증 후보\n\n아직 조건을 통과한 후보가 없습니다."
    lines = [
        "## 거래량/거래대금 급증 후보",
        "",
        "| 종목 | 점수 | 현재가 | 거래량 배율 | 최근 거래대금 | 단기 수익률 | 당일 수익률 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in candidates[:limit]:
        lines.append(
            f"| {item['symbol']} | {float(item['score']):.2f} | {_money(float(item['price']))} | "
            f"{float(item['volume_ratio']):.2f}x | {_money(float(item['recent_dollar_volume']))} | "
            f"{float(item['short_return']):.2%} | {float(item['day_return']):.2%} |"
        )
    return "\n".join(lines)


def _write_realtime_equity_chart(path: Path, rows: list[dict[str, str]]) -> Path | None:
    if len(rows) < 2:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    _configure_korean_font()
    timestamps = [datetime.fromisoformat(row["timestamp"]) for row in rows]
    equities = [float(row["equity"]) for row in rows]
    fig, ax = plt.subplots(figsize=(12, 5.8), dpi=140)
    ax.plot(timestamps, equities, color="#2563eb", linewidth=2.2)
    ax.set_title("실시간 paper trading 총자산", fontsize=15, fontweight="bold", loc="left")
    ax.set_ylabel("총자산($)")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _configure_korean_font() -> None:
    try:
        import matplotlib
        from matplotlib import font_manager

        available = {font.name for font in font_manager.fontManager.ttflist}
        candidates = ["Malgun Gothic", "Noto Sans KR", "Gulim", "Batang", "AppleGothic"]
        selected = next((font for font in candidates if font in available), "DejaVu Sans")
        matplotlib.rcParams["font.family"] = selected
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        return


def _max_drawdown(equities: list[float]) -> float:
    peak = 0.0
    worst = 0.0
    for value in equities:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1)
    return worst


def _tick_spread_pct(tick: PriceTick) -> float:
    if tick.bid is None or tick.ask is None or tick.price <= 0:
        return 0.0
    return max(tick.ask - tick.bid, 0.0) / tick.price


def _feed_scope(feed: str) -> str:
    normalized = feed.lower()
    if normalized == "iex":
        return "IEX single-venue feed"
    if normalized == "sip":
        return "SIP consolidated feed"
    return f"{feed} feed"


def _feed_limitation(feed: str) -> str:
    normalized = feed.lower()
    if normalized == "iex":
        return "does not represent full US consolidated volume/order flow"
    if normalized == "sip":
        return "requires account data entitlement; broader than IEX when available"
    return "feed coverage depends on provider entitlement"


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _short_time(value: str) -> str:
    if not value:
        return "-"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    return parsed.astimezone(KST).strftime("%m-%d %H:%M:%S")
